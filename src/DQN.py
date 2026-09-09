import numpy as np
import pandas as pd
import copy
import torch
from replay_buffer import PrioritizedReplayBuffer
from qfunction import Q_Fun
from mutation_frequency import (
    TOTAL_KIRC_TUMOR_SAMPLES,
    classify_mutation_frequency,
    mutation_frequency_pct,
)
from rd_definitions import (
    RD_EVIDENCE_CAP_DEFAULT,
    RD_EVIDENCE_MIN_DEFAULT,
    RD_EVIDENCE_SCALE_DEFAULT,
    RD_REWARD_WEIGHT_DEFAULTS,
    RD_W_DISCOVERY_DEFAULT,
    RD_W_RECOVERY_DEFAULT,
)


# Legacy reward-scale magic numbers（改动会改变训练动态，务必先理解 reward 设计）：
#   get_reward:  weight_score = weight_sum * n_actions / 150   （150 = 历史 topk）
#   step:        patient component = patient_improve * 5.0，整体再乘 (1 - score_alpha) * 3000
#                score component      = score_improve * 0.01
#                driver ratio         = driver_improve * 5.0
#                per-gene driver bonus = train_label_bonus（默认 1.0，可配置）
#                reward 截断到 [0, 5.0]
#   learn:       (修复 2026-09-01)reward 不再除以 10.0；此前该缩放使 reward 占 TD target <1%，
#                TD 目标退化为 gamma*Q_next 的自洽平滑，奖励信号被淹没，详见 README「RL 学习回路审计」。
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            net_ori,
            fea_ori,
            embedding_size,
            train_patient_data,
            test_patient_data,
            gene_sta,
            weights,
            score_alpha,
            train_driver_set=None,
            pat_num=0,
            learning_rate=0.0001,
            reward_decay=0.95,
            memory_size=50000,
            batch_size=128,
            selection_budget=999,
            gradient_clip=1.0,
            reward_mode="legacy",
            reward_weights=None,
            reward_feature_columns=None,
            lowfreq_evidence_by_gene=None,
            preference_dim=0,
            q_preference_dim=None,
            objective_dim=1,
            vector_value_normalization="legacy_loss_scale",
            vector_action_scalarization="normalized",
            learning_mode="ddqn",
    ):
        self.fea_ori = copy.deepcopy(fea_ori)
        self.train_patient_data = train_patient_data
        self.test_patient_data = test_patient_data
        self.weights = weights
        # Reward driver labels. gene_sta is kept as a backwards-compatible alias.
        self.train_driver_set = set(train_driver_set if train_driver_set is not None else gene_sta)
        self.gene_sta = self.train_driver_set
        self.actions = []
        self.actions_index = np.ones(n_actions)
        self.n_actions = n_actions
        self.feature_dim = self.fea_ori.shape[1] if len(self.fea_ori.shape) == 2 else 3
        self.preference_dim = int(preference_dim)
        if self.preference_dim < 0:
            raise ValueError(f"preference_dim must be non-negative, got {preference_dim!r}")
        self.q_preference_dim = self.preference_dim if q_preference_dim is None else int(q_preference_dim)
        if self.q_preference_dim < 0:
            raise ValueError(f"q_preference_dim must be non-negative, got {q_preference_dim!r}")
        self.objective_dim = int(objective_dim)
        if self.objective_dim not in {1, 2}:
            raise ValueError(f"objective_dim must be 1 or 2, got {objective_dim!r}")
        if self.objective_dim == 2 and self.q_preference_dim:
            raise ValueError("Vector-Q must not condition its heads on preference; scalarize only at action selection.")
        self.vector_value_normalization = str(vector_value_normalization)
        if self.objective_dim == 2 and self.vector_value_normalization not in {
                "legacy_loss_scale", "popart"
        }:
            raise ValueError(
                "vector_value_normalization must be 'legacy_loss_scale' or 'popart'."
            )
        self.vector_action_scalarization = str(vector_action_scalarization)
        if self.vector_action_scalarization not in {"normalized", "raw"}:
            raise ValueError("vector_action_scalarization must be 'normalized' or 'raw'.")
        self.learning_mode = str(learning_mode)
        if self.learning_mode not in {"ddqn", "contextual_bandit"}:
            raise ValueError("learning_mode must be 'ddqn' or 'contextual_bandit'.")
        if self.learning_mode == "contextual_bandit" and self.objective_dim == 2:
            if self.preference_dim != 2 or self.q_preference_dim != 0:
                raise ValueError(
                    "Dual-head contextual bandit requires preference_dim=2 and q_preference_dim=0; "
                    "preferences are used only to combine the two action-value heads."
                )
        self.embedding_size = embedding_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_min = 0.01
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.selection_budget = int(selection_budget)
        if self.selection_budget <= 0:
            raise ValueError(f"selection_budget must be positive, got {selection_budget!r}")
        self.epsilon = 1.0
        self.gradient_clip = float(gradient_clip)
        if self.gradient_clip <= 0:
            raise ValueError(f"gradient_clip must be positive, got {gradient_clip!r}")
        self.score_be = 0
        self.score_sta = 0
        self.score_pat = 0
        self.learn_step_counter = 0
        self.objective_value_scale = np.ones(self.objective_dim, dtype=np.float32)
        self.objective_value_mean = np.zeros(self.objective_dim, dtype=np.float32)
        self.objective_scale_decay = 0.99

        self.memory_counter = 0
        self.tau = 0.001

        self.reward_all = 0
        self.reward_list = []
        self.last_reward_components = {}
        self.last_learn_metrics = {}
        self.reward_mode = reward_mode
        self.reward_weights = dict(reward_weights or {})
        self.reward_feature_columns = list(reward_feature_columns or [])
        self.reward_feature_percentiles = self._build_reward_feature_percentiles()
        self.lowfreq_evidence_by_gene = {
            self._clean_reward_gene(gene): dict(values)
            for gene, values in (lowfreq_evidence_by_gene or {}).items()
        }
        self.lowfreq_unlabeled_score_by_mode = {
            "lowfreq_unlabeled_evidence": "LowFrequencyEvidenceScore",
            "lowfreq_unlabeled_evidence_v2": "LowFrequencyEvidenceScoreV2",
            "lowfreq_unlabeled_no_network": "EvidenceWithoutNetwork",
            "lowfreq_unlabeled_no_omics": "EvidenceWithoutOmics",
            "lowfreq_unlabeled_no_rarity": "EvidenceWithoutRarity",
        }
        if (
                self.reward_mode in self.lowfreq_unlabeled_score_by_mode
                and not self.lowfreq_evidence_by_gene
        ):
            raise ValueError(
                f"{self.reward_mode} requires a frozen low-frequency evidence table."
            )
        if (
                self.reward_mode == "rd_scan"
                and not self.lowfreq_evidence_by_gene
        ):
            raise ValueError(
                "rd_scan requires a frozen low-frequency evidence table "
                "(--lowfreq-evidence-path) for the discovery (evidenceV2) signal."
            )
        T = 3
        ALPHA = 0.0001
        self.Q = Q_Fun(self.feature_dim, self.embedding_size, T, ALPHA, net_ori, self.q_preference_dim, self.objective_dim)
        self.Q_target = Q_Fun(self.feature_dim, self.embedding_size, T, ALPHA, net_ori, self.q_preference_dim, self.objective_dim)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()
        print(f"📌 Q network input feature dim: {self.feature_dim}")
        print(f"📌 Selection budget: {self.selection_budget}")
        dtype_bytes = np.dtype(np.float32).itemsize
        shared_state_bytes = self.n_actions * self.feature_dim * dtype_bytes
        mask_bytes = 2 * self.memory_size * self.n_actions * dtype_bytes
        gib = 1024 ** 3
        print(
            "📌 Replay buffer memory estimate before allocation: "
            f"memory_size={self.memory_size}, n_actions={self.n_actions}, "
            f"feature_dim={self.feature_dim}, shared_state≈{shared_state_bytes / gib:.4f} GiB, "
            f"action_masks≈{mask_bytes / gib:.3f} GiB "
            "(excluding action, reward, done, priority arrays and runtime overhead)"
        )
        if mask_bytes > 4 * gib:
            print(
                "⚠️ 当前经验池可能占用大量内存，请通过 --memory-size 设置较小容量。"
            )
        self.memory = PrioritizedReplayBuffer(
            self.memory_size,
            self.n_actions,
            feature_dim=self.feature_dim,
            alpha=0.2,
            beta_start=0.1,
            beta_frames=2000000,
            eps=1e-5,
            preference_dim=self.preference_dim,
            reward_dim=self.objective_dim,
        )

        self.score_alpha = score_alpha
        self.pat_num = pat_num
        self.embedding = None

        for name, param in self.Q.named_parameters():
            pass

    def _build_reward_feature_percentiles(self):
        percentiles = {}
        if self.fea_ori.ndim != 2:
            return percentiles
        for column in ("Mutation", "Expression", "Methylation"):
            if column not in self.reward_feature_columns:
                continue
            idx = self.reward_feature_columns.index(column)
            values = np.asarray(self.fea_ori[:, idx], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{column} reward feature contains NaN or Inf.")
            ranked = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)
            ranked = np.clip(ranked, 0.0, 1.0)
            percentiles[column] = ranked
        return percentiles

    @staticmethod
    def default_reward_weights():
        weights = {
            "multiomics_mutation_weight": 0.08,
            "multiomics_expression_weight": 0.06,
            "multiomics_methylation_weight": 0.06,
            "no_mutation_expression_weight": 0.08,
            "no_mutation_methylation_weight": 0.08,
            "lowfreq_expression_weight": 0.05,
            "lowfreq_methylation_weight": 0.05,
            "lowfreq_bonus_cap": 0.20,
            "lowfreq_unlabeled_bonus_scale": 0.0,
            "lowfreq_unlabeled_bonus_cap": 0.0,
            "train_label_bonus": 1.0,
            "w_recovery": RD_W_RECOVERY_DEFAULT,
            "w_discovery": RD_W_DISCOVERY_DEFAULT,
            "rd_evidence_min": RD_EVIDENCE_MIN_DEFAULT,
            "rd_evidence_scale": RD_EVIDENCE_SCALE_DEFAULT,
            "rd_evidence_cap": RD_EVIDENCE_CAP_DEFAULT,
        }
        return {**RD_REWARD_WEIGHT_DEFAULTS, **weights}

    @staticmethod
    def _clean_reward_gene(gene):
        if gene is None:
            return None
        gene = str(gene).strip().upper()
        if "|" in gene:
            gene = gene.split("|", 1)[0].strip()
        return gene

    def reward_config(self):
        weights = self.default_reward_weights()
        weights.update(self.reward_weights)
        return {
            "reward_mode": self.reward_mode,
            "reward_weights": weights,
            "reward_bounds": [0.0, 5.0],
            "percentile_features": sorted(self.reward_feature_percentiles),
            "lowfreq_evidence_gene_count": len(self.lowfreq_evidence_by_gene),
            "lowfreq_unlabeled_score_modes": dict(self.lowfreq_unlabeled_score_by_mode),
            "validation_labels_used_for_reward": False,
        }

    def _reward_weight(self, name):
        weights = self.default_reward_weights()
        weights.update(self.reward_weights)
        return float(weights[name])

    def set_preference(self, preference):
        """Activate a Recovery–Discovery preference for the next collected episode.

        The scalar reward definition remains exactly ``rd_scan``; this method only
        supplies its two existing weights and keeps the same vector available to
        the preference-conditioned Q network/replay transition.
        """
        if self.preference_dim != 2:
            raise RuntimeError("set_preference requires preference_dim=2.")
        value = np.asarray(preference, dtype=np.float32)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"preference must be a finite shape-(2,) vector, got {preference!r}")
        if np.any(value < 0.0) or not np.isclose(float(value.sum()), 1.0, atol=1e-6):
            raise ValueError("preference must be non-negative and sum to 1.")
        if self.reward_mode != "rd_scan":
            raise RuntimeError("Preference conditioning is only valid for reward_mode='rd_scan'.")
        # Scalar rd_scan logging keeps its historical weighted reward.  Vector-Q
        # learning reads the two raw components recorded by step() instead.
        self.reward_weights["w_recovery"] = float(value[0])
        self.reward_weights["w_discovery"] = float(value[1])
        self.active_preference = value.copy()
        return self.active_preference

    def scalarize_q(self, q_values, preference):
        """Combine vector action values under the configured preference."""
        if self.objective_dim == 1:
            return q_values
        pref = torch.as_tensor(preference, dtype=q_values.dtype, device=q_values.device)
        expected = (2,) if q_values.ndim == 2 else (q_values.shape[0], 2)
        if tuple(pref.shape) != expected:
            raise ValueError(f"preference shape must be {expected}, got {tuple(pref.shape)}")
        if self.vector_action_scalarization == "raw":
            values = self.denormalize_objective_q(q_values)
        elif self.vector_value_normalization == "popart":
            # PopArt heads already predict z=(Q_raw-mean)/scale.
            values = q_values
        else:
            scale = torch.as_tensor(self.objective_value_scale, dtype=q_values.dtype, device=q_values.device)
            values = q_values / scale
        return torch.sum(values * (pref if q_values.ndim == 2 else pref.unsqueeze(1)), dim=-1)

    def denormalize_objective_q(self, q_values):
        """Map PopArt normalized head values back to raw discounted returns."""
        if self.objective_dim != 2 or self.vector_value_normalization != "popart":
            return q_values
        mean = torch.as_tensor(self.objective_value_mean, dtype=q_values.dtype, device=q_values.device)
        scale = torch.as_tensor(self.objective_value_scale, dtype=q_values.dtype, device=q_values.device)
        return q_values * scale + mean

    def normalize_objective_q(self, q_values):
        if self.objective_dim != 2 or self.vector_value_normalization != "popart":
            return q_values
        mean = torch.as_tensor(self.objective_value_mean, dtype=q_values.dtype, device=q_values.device)
        scale = torch.as_tensor(self.objective_value_scale, dtype=q_values.dtype, device=q_values.device)
        return (q_values - mean) / scale

    @staticmethod
    def _rescale_popart_output_head(model, old_mean, old_scale, new_mean, new_scale):
        """Change normalized output coordinates without changing raw Q values."""
        layer = model.lin8
        with torch.no_grad():
            old_mean = torch.as_tensor(old_mean, dtype=layer.bias.dtype, device=layer.bias.device)
            old_scale = torch.as_tensor(old_scale, dtype=layer.bias.dtype, device=layer.bias.device)
            new_mean = torch.as_tensor(new_mean, dtype=layer.bias.dtype, device=layer.bias.device)
            new_scale = torch.as_tensor(new_scale, dtype=layer.bias.dtype, device=layer.bias.device)
            factor = old_scale / new_scale
            layer.weight.mul_(factor.unsqueeze(1))
            layer.bias.copy_((old_scale * layer.bias + old_mean - new_mean) / new_scale)

    def _update_popart_statistics(self, raw_targets):
        """EMA PopArt statistics and output-head rescaling for both Q networks."""
        if self.objective_dim != 2 or self.vector_value_normalization != "popart":
            return
        values = raw_targets.detach().to(dtype=torch.float32)
        batch_mean = values.mean(dim=0).cpu().numpy()
        batch_second = values.square().mean(dim=0).cpu().numpy()
        old_mean = self.objective_value_mean.copy()
        old_scale = self.objective_value_scale.copy()
        old_second = np.square(old_scale) + np.square(old_mean)
        new_mean = self.objective_scale_decay * old_mean + (1.0 - self.objective_scale_decay) * batch_mean
        new_second = self.objective_scale_decay * old_second + (1.0 - self.objective_scale_decay) * batch_second
        new_scale = np.sqrt(np.maximum(new_second - np.square(new_mean), 1e-6)).astype(np.float32)
        self._rescale_popart_output_head(self.Q, old_mean, old_scale, new_mean, new_scale)
        self._rescale_popart_output_head(self.Q_target, old_mean, old_scale, new_mean, new_scale)
        self.objective_value_mean = new_mean.astype(np.float32)
        self.objective_value_scale = new_scale

    def vector_reward(self):
        if self.objective_dim != 2:
            raise RuntimeError("vector_reward is only valid for objective_dim=2.")
        value = np.asarray([
            self.last_reward_components.get("reward_recovery_raw"),
            self.last_reward_components.get("reward_discovery_raw"),
        ], dtype=np.float32)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise RuntimeError("rd_scan did not produce finite raw Recovery/Discovery rewards.")
        return value

    def _feature_percentile(self, column, action):
        values = self.reward_feature_percentiles.get(column)
        if values is None:
            return 0.0
        return float(values[action])

    def _zero_components(self):
        return {
            "reward_total": 0.0,
            "reward_legacy": 0.0,
            "reward_train_label": 0.0,
            "reward_mutation": 0.0,
            "reward_expression": 0.0,
            "reward_methylation": 0.0,
            "reward_lowfreq": 0.0,
            "reward_evidence_bonus": 0.0,
            "reward_penalty": 0.0,
        }

    def _lowfreq_evidence_for_gene(self, gene):
        return self.lowfreq_evidence_by_gene.get(
            self._clean_reward_gene(gene),
            {
                "MutationPatientCount": 0.0,
                "MutationFrequency": 0.0,
                "MutationFrequencyPct": 0.0,
                "MutationGroup": "very_low",
                "MutationRarityScore": 0.0,
                "NonMutationOmicsSupport": 0.0,
                "DegreeCorrectedNetworkSupport": 0.0,
                "LowFrequencyEvidenceScore": 0.0,
                "EvidenceWithoutRarity": 0.0,
                "EvidenceWithoutOmics": 0.0,
                "EvidenceWithoutNetwork": 0.0,
                "CNVFunctionalSupport": 0.0,
                "ExpressionSupport": 0.0,
                "MethylationSupport": 0.0,
                "CNVAvailable": False,
            },
        )

    def _compose_reward_components(
            self,
            action,
            selected_gene,
            is_train_driver,
            legacy_base,
            train_label_bonus,
    ):
        components = self._zero_components()
        components["reward_legacy"] = float(legacy_base)
        components["reward_train_label"] = float(train_label_bonus)

        mode = self.reward_mode
        if mode == "legacy":
            pass
        elif mode == "multiomics_mutation":
            components["reward_mutation"] = (
                self._reward_weight("multiomics_mutation_weight")
                * self._feature_percentile("Mutation", action)
            )
            components["reward_expression"] = (
                self._reward_weight("multiomics_expression_weight")
                * self._feature_percentile("Expression", action)
            )
            components["reward_methylation"] = (
                self._reward_weight("multiomics_methylation_weight")
                * self._feature_percentile("Methylation", action)
            )
        elif mode == "multiomics_no_mutation":
            components["reward_mutation"] = 0.0
            components["reward_expression"] = (
                self._reward_weight("no_mutation_expression_weight")
                * self._feature_percentile("Expression", action)
            )
            components["reward_methylation"] = (
                self._reward_weight("no_mutation_methylation_weight")
                * self._feature_percentile("Methylation", action)
            )
        elif mode in {"multiomics_lowfreq", "label_conditioned_lowfreq"}:
            components["reward_mutation"] = 0.0
            components["reward_expression"] = (
                self._reward_weight("lowfreq_expression_weight")
                * self._feature_percentile("Expression", action)
            )
            components["reward_methylation"] = (
                self._reward_weight("lowfreq_methylation_weight")
                * self._feature_percentile("Methylation", action)
            )
            mutation_percentile = self._feature_percentile("Mutation", action)
            rarity = float(np.clip(1.0 - mutation_percentile, 0.0, 1.0))
            if is_train_driver:
                components["reward_lowfreq"] = self._reward_weight("lowfreq_bonus_cap") * rarity
        elif mode in self.lowfreq_unlabeled_score_by_mode:
            evidence = self._lowfreq_evidence_for_gene(selected_gene)
            score_column = self.lowfreq_unlabeled_score_by_mode[mode]
            score = float(evidence.get(score_column, 0.0))
            score = float(np.clip(score, 0.0, 1.0))
            if not is_train_driver:
                raw_bonus = self._reward_weight("lowfreq_unlabeled_bonus_scale") * score
                components["reward_evidence_bonus"] = float(
                    np.clip(raw_bonus, 0.0, self._reward_weight("lowfreq_unlabeled_bonus_cap"))
                )
        elif mode == "rd_scan":
            # 混合奖励：reward = w_recovery × recovery(legacy 恢复信号)
            #            + w_discovery × discovery(低频新候选 evidenceV2 信号)。
            # components 记录“加权贡献”，保证 sum(components)==raw_total 的审计成立：
            #   reward_legacy/  reward_train_label  = w_recovery × legacy 原值
            #   reward_evidence_bonus = w_discovery × discovery 原值
            w_rec = self._reward_weight("w_recovery")
            w_disc = self._reward_weight("w_discovery")
            evidence = self._lowfreq_evidence_for_gene(selected_gene)
            score = float(np.clip(float(evidence.get("LowFrequencyEvidenceScoreV2", 0.0)), 0.0, 1.0))
            mutation_count = int(round(float(evidence.get("MutationPatientCount", 0.0))))
            discovery_raw = 0.0
            # discovery 门控（frozen-label：只用低频区 + 证据分 + 非 train driver，不读 val）
            if (
                    not is_train_driver
                    and classify_mutation_frequency(mutation_count) == "low_frequency"
                    and score >= self._reward_weight("rd_evidence_min")
            ):
                discovery_raw = float(np.clip(
                    self._reward_weight("rd_evidence_scale") * score,
                    0.0,
                    self._reward_weight("rd_evidence_cap"),
                ))
            components["reward_legacy"] = float(w_rec * legacy_base)
            components["reward_train_label"] = float(w_rec * train_label_bonus)
            components["reward_evidence_bonus"] = float(w_disc * discovery_raw)
            # These are the *unweighted endpoint rewards* used by vector-Q.
            # Keeping the historical [0, 5] clip here makes vector head 0/1
            # exactly agree with the old scalar rd_scan reward at w=(1,0) and
            # w=(0,1), respectively.  The weighted scalar components above are
            # retained unchanged for legacy logging only.
            components["_recovery_raw"] = float(np.clip(legacy_base + train_label_bonus, 0.0, 5.0))
            components["_discovery_raw"] = float(np.clip(discovery_raw, 0.0, 5.0))
        else:
            raise ValueError(f"Unsupported reward_mode: {mode!r}")

        raw_total = sum(
            value for key, value in components.items()
            if key != "reward_total" and not key.startswith("_")
        )
        total = float(np.clip(raw_total, 0.0, 5.0))
        components["reward_penalty"] += total - raw_total
        components["reward_total"] = total

        if not np.isfinite(list(components.values())).all():
            raise FloatingPointError(f"Non-finite reward components for {selected_gene}: {components}")
        return components

    def get_reward(self, gene_num, gene_name):
        """
        计算当前已选择基因集合的三个累计指标：

        1. weight_score:
           基因权重累计得分，保留原始代码逻辑。

        2. patient_uncovered_score:
           未覆盖患者比例，保留原始代码逻辑。
           注意：原始代码使用的是未覆盖患者比例，数值越小代表覆盖越多。

        3. driver_hit_ratio:
           当前已选择基因中，命中 ccRCC 已知癌症相关 / driver gene 的比例。
           这个指标越高，说明当前子图越靠近已知 ccRCC 癌症基因集合。

        重要：
        未知基因不等于负样本，因此不对未知基因直接扣分。
        """

        weight_sum = 0
        gene_name = list(gene_name)
        patient_num = []
        driver_hit_num = 0

        for i in self.actions:
            gene = gene_name[i]

            # 统计命中的训练 driver gene 数量
            if gene in self.train_driver_set:
                driver_hit_num += 1

            # 保留原始权重和患者覆盖逻辑
            if gene not in gene_num:
                weight_sum += self.weights[gene]
            else:
                patient_num.extend(gene_num[gene])
                weight_sum += self.weights[gene]

        if len(self.actions) == 0:
            driver_hit_ratio = 0.0
        else:
            driver_hit_ratio = driver_hit_num / len(self.actions)

        weight_score = weight_sum * self.n_actions / 150
        patient_uncovered_score = (self.pat_num - len(set(patient_num))) / self.pat_num

        return weight_score, patient_uncovered_score, driver_hit_ratio

    def step(self, network, action, gene_num, gene_name, weights):
        """
        ccRCC 癌症相关 / driver gene 预测的温和 reward 版本。

        修改目的：
        1. 防止累计 reward 突然暴涨；
        2. 降低患者覆盖项和综合分数项的放大倍数；
        3. 保留 driver gene 命中奖励；
        4. 未知基因不扣分，避免抑制新候选基因发现。
        """

        actions = self.actions[:]

        weight_score, patient_uncovered_score, driver_hit_ratio = self.get_reward(
            gene_num,
            gene_name
        )

        score_new = (
                self.score_alpha * weight_score
                + (1 - self.score_alpha) * patient_uncovered_score * 3000
        )

        legacy_base = 0.0

        # ========== 1. 患者覆盖改进奖励 ==========
        # patient_uncovered_score 越小越好
        patient_improve = self.score_pat - patient_uncovered_score

        if self.score_pat == 0:
            patient_improve = 0.0

        patient_coverage_component = patient_improve * 5.0 if patient_improve > 0 else 0.0
        if patient_improve > 0:
            legacy_base += patient_coverage_component

        # ========== 2. 综合分数改进奖励 ==========
        # score_new 越小越好
        score_improve = self.score_be - score_new

        if self.score_be == 0:
            score_improve = 0.0

        score_component = score_improve * 0.01 if score_improve > 0 else 0.0
        if score_improve > 0:
            legacy_base += score_component

        # ========== 3. driver gene 比例提升奖励 ==========
        driver_improve = driver_hit_ratio - self.score_sta

        driver_ratio_component = driver_improve * 5.0 if driver_improve > 0 else 0.0
        if driver_improve > 0:
            legacy_base += driver_ratio_component

        # ========== 4. 当前动作直接命中奖励 ==========
        selected_gene = list(gene_name)[action]
        is_train_driver = selected_gene in self.train_driver_set
        driver_label_bonus = self._reward_weight("train_label_bonus") if is_train_driver else 0.0

        if is_train_driver:
            pass

        # ========== 5. reward 裁剪，防止异常暴涨 ==========
        components = self._compose_reward_components(
            action=action,
            selected_gene=selected_gene,
            is_train_driver=is_train_driver,
            legacy_base=legacy_base,
            train_label_bonus=driver_label_bonus,
        )
        reward = float(components["reward_total"])
        evidence = self._lowfreq_evidence_for_gene(selected_gene)
        mutation_count = int(round(float(evidence.get("MutationPatientCount", 0.0))))
        mutation_group = evidence.get("MutationGroup") or classify_mutation_frequency(mutation_count)
        base_reward = float(np.clip(legacy_base + driver_label_bonus, 0.0, 5.0))
        self.last_reward_components = {
            "selected_gene": selected_gene,
            "is_train_driver": bool(is_train_driver),
            **components,
            "reward_recovery_raw": float(components.get("_recovery_raw", base_reward)),
            "reward_discovery_raw": float(components.get("_discovery_raw", 0.0)),
            "base_reward": base_reward,
            "evidence_bonus": float(components.get("reward_evidence_bonus", 0.0)),
            "final_reward": float(reward),
            "driver_label_bonus": float(driver_label_bonus),
            "weight_component": float(weight_score),
            "patient_coverage_component": float(patient_coverage_component),
            "score_component": float(score_component),
            "driver_ratio_component": float(driver_ratio_component),
            "unclipped_reward": float(legacy_base + driver_label_bonus),
            "clipped_reward": float(reward),
            "mutation_percentile": self._feature_percentile("Mutation", action),
            "expression_percentile": self._feature_percentile("Expression", action),
            "methylation_percentile": self._feature_percentile("Methylation", action),
            "rarity_score": float(np.clip(1.0 - self._feature_percentile("Mutation", action), 0.0, 1.0)),
            "mutation_count": float(mutation_count),
            "mutation_frequency": float(evidence.get("MutationFrequency", 0.0)),
            "MutationFrequencyPct": float(
                evidence.get("MutationFrequencyPct", mutation_frequency_pct(mutation_count, TOTAL_KIRC_TUMOR_SAMPLES))
            ),
            "MutationGroup": str(mutation_group),
            "MutationRarityScore": float(evidence.get("MutationRarityScore", 0.0)),
            "ExpressionSupport": float(evidence.get("ExpressionSupport", 0.0)),
            "MethylationSupport": float(evidence.get("MethylationSupport", 0.0)),
            "CNVFunctionalSupport": float(evidence.get("CNVFunctionalSupport", 0.0)),
            "NonMutationOmicsSupport": float(evidence.get("NonMutationOmicsSupport", 0.0)),
            "DegreeCorrectedNetworkSupport": float(
                evidence.get("DegreeCorrectedNetworkSupport", evidence.get("DegreeCorrectedNetworkSupportV2", 0.0))
            ),
            "DegreeCorrectedNetworkSupportV2": float(evidence.get("DegreeCorrectedNetworkSupportV2", 0.0)),
            "LowFrequencyEvidenceScore": float(
                evidence.get("LowFrequencyEvidenceScore", evidence.get("LowFrequencyEvidenceScoreV2", 0.0))
            ),
            "LowFrequencyEvidenceScoreV2": float(evidence.get("LowFrequencyEvidenceScoreV2", 0.0)),
            "EvidenceWithoutRarity": float(evidence.get("EvidenceWithoutRarity", evidence.get("EvidenceWithoutRarityV2", 0.0))),
            "EvidenceWithoutRarityV2": float(evidence.get("EvidenceWithoutRarityV2", 0.0)),
            "EvidenceWithoutOmics": float(evidence.get("EvidenceWithoutOmics", evidence.get("EvidenceWithoutOmicsV2", 0.0))),
            "EvidenceWithoutOmicsV2": float(evidence.get("EvidenceWithoutOmicsV2", 0.0)),
            "EvidenceWithoutNetwork": float(evidence.get("EvidenceWithoutNetwork", evidence.get("EvidenceWithoutNetworkV2", 0.0))),
            "EvidenceWithoutNetworkV2": float(evidence.get("EvidenceWithoutNetworkV2", 0.0)),
            "CNVAvailable": bool(evidence.get("CNVAvailable", False)),
            "reward_mode": self.reward_mode,
        }

        self.reward_all += reward

        self.score_be = score_new
        self.score_sta = driver_hit_ratio
        self.score_pat = patient_uncovered_score

        done = 0

        if len(actions) >= self.selection_budget:
            self.embedding = None
            done = 1

            self.actions = []
            self.reward_list.append(reward)
            self.reward_all = 0

        return reward, done, actions

    def remember(self, *args, preference=None, history_mask=None, next_history_mask=None):
        self.memory.store_transition(
            *args,
            preference=preference,
            history_mask=history_mask,
            next_history_mask=next_history_mask,
        )
        self.memory_counter += 1

    def clear_mem(self):
        self.memory.clear()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        self.Q.train()
        # The target network is a deterministic evaluator. eval() disables dropout
        # while gradients remain disabled in the target-value block below.
        self.Q_target.eval()
        # ========== 从经验回放池采样一批经验 ==========
        # 对应：(S, A, R, S', 动作掩码)
        state, action, reward_sum, action_index, next_action_index, history_index, next_history_index, done, sample_indices, is_weights, preference = \
            self.memory.sample_buffer(self.batch_size)
        # 清空 Q 网络的梯度（上一步的梯度残留要清掉）
        self.Q.optimizer.zero_grad()
        mu = None

        state = torch.tensor(state, dtype=torch.float32).to(self.Q.device)
        action = torch.LongTensor(action).view(-1, 1).to(self.Q.device)
        reward_sum = torch.tensor(reward_sum, dtype=torch.float32).view(-1, self.objective_dim).to(self.Q.device)
        # 修复(2026-09-01):不再除以 10.0。此前 reward/10 使 TD target 中 reward 占比极低
        # (legacy 每步均值 0.012 → /10 后 0.0012,而 Q_next 约 1~5,reward 贡献 <1%),
        # 目标几乎退化为 gamma*Q_next 的价值自洽,奖励信号被完全淹没。
        new_state = state.clone()

        action_index = torch.LongTensor(action_index).to(self.Q.device)
        next_action_index = torch.LongTensor(next_action_index).to(self.Q.device)
        history_index = torch.LongTensor(history_index).to(self.Q.device)
        next_history_index = torch.LongTensor(next_history_index).to(self.Q.device)
        done = torch.tensor(done, dtype=torch.float32).view(-1, 1).to(self.Q.device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.Q.device)
        preference_tensor = (
            torch.tensor(preference, dtype=torch.float32).to(self.Q.device)
            if self.q_preference_dim else None
        )

                # ========== 计算目标 Q 值 y_t ==========
        # DDQN bootstraps from Q(s',a*).  The contextual-bandit control keeps
        # the identical state/action encoder, replay schedule and reward, but
        # learns only the immediate action reward Q(s,a)≈r(s,a).
        if self.learning_mode == "contextual_bandit":
            y_target = reward_sum
        else:
            # ========== 计算目标 Q 值 y_t (Double DQN 升级版) ==========
            with torch.no_grad():  # 眺望未来不需要计算梯度，省显存+加速
                # Online Network 负责动作选择：a* = argmax Q_online(s'_i, a')
                next_q_values_online, _ = self.Q(
                    mu, new_state, next_history_index, batch_flag=True, preference=preference_tensor
                )
                mask = next_action_index == 0
                action_scores = (
                    next_q_values_online if self.objective_dim == 1 else self.scalarize_q(
                        next_q_values_online,
                        torch.tensor(preference, dtype=torch.float32, device=self.Q.device),
                    )
                )
                best_next_actions = action_scores.masked_fill(mask, -1e9).argmax(dim=1, keepdim=True)

                # 第二步：目标网络 (self.Q_target) 当“裁判”，对 S' 进行独立打分
                next_q_values_target, _ = self.Q_target(
                    mu, new_state, next_history_index, batch_flag=True, preference=preference_tensor
                )
                next_q_target = (
                    next_q_values_target.gather(1, best_next_actions)
                    if self.objective_dim == 1 else next_q_values_target.gather(
                        1, best_next_actions.unsqueeze(-1).expand(-1, 1, self.objective_dim)
                    ).squeeze(1)
                )
                next_q_target = self.denormalize_objective_q(next_q_target)
                y_target = reward_sum + self.gamma * (1.0 - done) * next_q_target

        # PopArt updates the coordinate system before the gradient forward pass.
        # Both online and target lin8 heads are rescaled so their *raw* Q values
        # remain continuous while the network learns standardized TD targets.
        if self.objective_dim == 2 and self.vector_value_normalization == "popart":
            self._update_popart_statistics(y_target)

        # ========== 计算当前 Q 值 Q(s,a;θ) ==========
        y_pred_all, _ = self.Q(
            mu,
            state,
            history_index,
            batch_flag=True,
            preference=preference_tensor,
        )# 用 Q 网络计算当前状态 S 的所有 Q 值
        y_pred = (
            y_pred_all.gather(1, action)
            if self.objective_dim == 1 else y_pred_all.gather(
                1, action.unsqueeze(-1).expand(-1, 1, self.objective_dim)
            ).squeeze(1)
        )
        # ========== 计算损失函数 Loss  ==========
        # 对应论文公式：L = (y_t - Q(s,a;θ))²
        y_pred_raw = self.denormalize_objective_q(y_pred)
        td_errors = y_target - y_pred_raw

        if self.objective_dim == 2:
            if self.vector_value_normalization == "popart":
                loss_prediction = y_pred
                loss_target = self.normalize_objective_q(y_target)
            else:
                target_abs = y_target.detach().abs().mean(dim=0).cpu().numpy()
                self.objective_value_scale = (
                    self.objective_scale_decay * self.objective_value_scale
                    + (1.0 - self.objective_scale_decay) * np.maximum(target_abs, 1e-3)
                ).astype(np.float32)
                scale = torch.as_tensor(self.objective_value_scale, dtype=torch.float32, device=self.Q.device)
                loss_prediction = y_pred / scale
                loss_target = y_target / scale
            elementwise_loss = torch.nn.functional.smooth_l1_loss(
                loss_prediction, loss_target, reduction="none"
            )
            loss = torch.mean(is_weights * elementwise_loss.mean(dim=1, keepdim=True))
        else:
            elementwise_loss = torch.nn.functional.smooth_l1_loss(y_pred, y_target, reduction="none")
            loss = torch.mean(is_weights * elementwise_loss)
        # ==========  反向传播更新 Q 网络权重 ==========
        loss.backward()
        # 🛡️ 新增：梯度裁剪防弹衣！强行把超过 1.0 的极端梯度削平，防止网络崩溃
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.Q.parameters(),
            max_norm=self.gradient_clip,
        )
        self.Q.optimizer.step()
        td_errors_np = td_errors.detach().abs().cpu().numpy()
        priority_errors = (
            td_errors_np.mean(axis=1, keepdims=True) / float(self.objective_value_scale.mean())
            if self.objective_dim == 2 else td_errors_np
        )
        self.memory.update_priorities(sample_indices, priority_errors)
        self.last_learn_metrics = {
            "loss": float(loss.item()),
            "td_error_abs_mean": float(np.mean(td_errors_np)),
            "gradient_norm": float(gradient_norm.item() if hasattr(gradient_norm, "item") else gradient_norm),
            "objective_value_mean": self.objective_value_mean.tolist(),
            "objective_value_scale": self.objective_value_scale.tolist(),
            "objective_td_error_abs_mean": td_errors_np.mean(axis=0).tolist(),
        }
        self.learn_step_counter += 1
        # The target network has no role in a contextual bandit. Keeping it
        # frozen makes the no-bootstrap contract explicit. DDQN retains its
        # original soft target update.
        if self.learning_mode == "ddqn":
            tau = self.tau
            for target_param, local_param in zip(self.Q_target.parameters(), self.Q.parameters()):
                target_param.data.copy_(
                    tau * local_param.data + (1.0 - tau) * target_param.data
                )


