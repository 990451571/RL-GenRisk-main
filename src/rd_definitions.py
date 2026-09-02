"""Recovery–Discovery 权重探针共享定义（2026-09-02）。

让「reward 侧的 discovery 门控」与「评估侧的两条 Discovery 指标」使用同一套
候选集定义，避免两侧漂移。全部只依赖 evidence 表与已知 driver 集，不读任何
Test/Val 之外的信息。

候选集定义（用户 2026-09-02 裁定：Discovery = 低频新候选，不能把所有低频基因都算）：
  - 低频区   = 2-18 个突变患者（mutation_frequency.low_frequency）
  - novel    = 低频区 且 不在 train∪val 已知 driver 集合
  - supported= novel 且 LowFrequencyEvidenceScoreV2 ≥ rd_evidence_min（默认 0.5，顶部 ~10%）

reward 侧门控（frozen-label 纪律：不读 Val 身份）：
  gene 处于低频区(2-18) 且 LowFrequencyEvidenceScoreV2 ≥ rd_evidence_min
  且 不是 train driver。不排除 val driver —— reward 不得使用 val 标签。
"""
from __future__ import annotations

from lowfreq_evidence import clean_gene
from mutation_frequency import (
    LOW_FREQUENCY_MUTATION_MAX_COUNT as RD_LOW_FREQ_MAX,
    LOW_FREQUENCY_MUTATION_MIN_COUNT as RD_LOW_FREQ_MIN,
)

# 候选集最小/最大突变患者数（低频区）
RD_LOW_FREQ_MIN_COUNT = RD_LOW_FREQ_MIN
RD_LOW_FREQ_MAX_COUNT = RD_LOW_FREQ_MAX

# evidenceV2 分数列与默认门控
RD_EVIDENCE_COLUMN = "LowFrequencyEvidenceScoreV2"
RD_EVIDENCE_MIN_DEFAULT = 0.5   # novel 证据分门限：默认取低频 novel 的顶部 ~10%
RD_EVIDENCE_SCALE_DEFAULT = 2.0  # reward：supported novel 单步 ≈ score×scale（≈1.0-1.2，与 label bonus 同量级）
RD_EVIDENCE_CAP_DEFAULT = 5.0    # discovery 单步 bonus 上限（与总 reward 截断一致）

# 权重默认值（单位权重时单独取一组与 legacy 完全等价）
RD_W_RECOVERY_DEFAULT = 1.0
RD_W_DISCOVERY_DEFAULT = 1.0

# 探针 5 组权重：(w_recovery, w_discovery)
RD_WEIGHT_GROUPS = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.5, 0.5),
    (0.2, 0.8),
    (0.0, 1.0),
]

# reward_weights 字典中这些键的默认值（并入 default_reward_weights 供 DQN 读取）
RD_REWARD_WEIGHT_DEFAULTS = {
    "w_recovery": RD_W_RECOVERY_DEFAULT,
    "w_discovery": RD_W_DISCOVERY_DEFAULT,
    "rd_evidence_min": RD_EVIDENCE_MIN_DEFAULT,
    "rd_evidence_scale": RD_EVIDENCE_SCALE_DEFAULT,
    "rd_evidence_cap": RD_EVIDENCE_CAP_DEFAULT,
}

# 参与 rd_scan 的 reward weight 键名清单（validate/build_agent 遍历用）
RD_REWARD_WEIGHT_KEYS = [
    "w_recovery",
    "w_discovery",
    "rd_evidence_min",
    "rd_evidence_scale",
    "rd_evidence_cap",
]


def in_low_freq_range(patient_count: int) -> bool:
    """突变患者数是否处于低频区 [2, 18]。"""
    return RD_LOW_FREQ_MIN_COUNT <= int(patient_count) <= RD_LOW_FREQ_MAX_COUNT


def discovery_sets_from_evidence(
    ev_df,
    known_genes,
    evidence_min: float = RD_EVIDENCE_MIN_DEFAULT,
    evidence_column: str = RD_EVIDENCE_COLUMN,
):
    """从 evidence 表构建三条 Discovery 候选集（评估用，已知集合排除 train∪val）。

    参数:
      ev_df:      读入的 evidence 表（含 MutationPatientCount / evidence_column 列）
      known_genes:可迭代的已知 driver（train∪val，评估侧需排除）
      evidence_min / evidence_column: supported 门控

    返回:
      dict {
        "lowfreq_novel":         低频区且非已知 driver 的基因集,
        "evidence_supported":    上述且证据分 ≥ evidence_min 的基因集,
        "n_lowfreq_novel": 计数,
        "n_evidence_supported": 计数,
      }
    """
    known = {clean_gene(g) for g in known_genes}
    known.discard(None)
    table = ev_df.copy()
    table["Gene"] = table["Gene"].map(clean_gene)
    table = table.dropna(subset=["Gene"])
    mask_lowfreq = table["MutationPatientCount"].between(
        RD_LOW_FREQ_MIN_COUNT, RD_LOW_FREQ_MAX_COUNT
    )
    sub = table.loc[mask_lowfreq & ~table["Gene"].isin(known)]
    lowfreq_novel = set(sub["Gene"])
    supported = set(sub.loc[sub[evidence_column] >= evidence_min, "Gene"])
    return {
        "lowfreq_novel": lowfreq_novel,
        "evidence_supported": supported,
        "n_lowfreq_novel": len(lowfreq_novel),
        "n_evidence_supported": len(supported),
    }
