# RL-GenRisk

基于多组学数据、生物网络、图神经网络与 DDQN/PER 强化学习的癌症相关基因排序研究代码库。

---

## 低频奖励实验记录（2026-08-31）

> ⚠️ **重要更正（2026-08-31）**：本节的单 seed（seed=0）结论——「最佳配置 = exp6（V2 + 去命中）」——已在后续**多种子验证**中被推翻：3 个种子下 V2+去命中并不稳定优于 legacy 基准，之前的领先属随机种子碰巧较好。详细结果见下文「多种子验证（2026-08-31）」一节。

### 一、目的

此前正式训练（100 轮 legacy）的验证结果较弱：NDCG@150 ≈ 0.06、Recall@150 ≈ 0.04（25 个验证 driver 基因只命中 1 个）。分析认为根因是：

1. **训练/验证基因不重叠** → reward 用 16 个 train driver，评估用 25 个不同的 val driver，「+1.0 直接命中训练标签」的奖励会诱导模型**背答案**而非学泛化规律；
2. **driver 信号太弱** → 被「患者覆盖」等信号淹没。

本实验验证两个假设：

- **假设 A**：低频证据（label 无关的冻结证据表，`lowfreq_unlabeled_*` reward）能否改善对低频 driver 基因的发现；
- **假设 B**：去掉「+1.0 直接命中训练标签」奖励，能否减少背答案、改善泛化。

### 二、计划（7 个 run，统一 seed 0、50 轮）

⚠️ 说明：这 7 个 run **不是 7 个不同的随机种子**，它们**都用同一个 seed = 0**；区别只在「reward 模式」和「train-label-bonus（直接命中奖励）」这两个变量。

| run | reward 模式 | train-label-bonus | 代表含义 |
|---|---|---|---|
| exp1 | legacy | 1.0（默认） | 基准对照（旧方法） |
| exp2 | lowfreq V1 | 1.0 | V1 证据 + 默认命中奖励 |
| exp3 | lowfreq V1 | 0.0 | V1 证据 + **去掉**命中奖励 |
| exp4 | lowfreq V1 | 0.5 | V1 证据 + 命中奖励**减半** |
| exp5 | lowfreq V2 | 1.0 | V2 证据 + 默认命中奖励 |
| exp6 | lowfreq V2 | 0.0 | V2 证据 + **去掉**命中奖励 |
| exp7 | lowfreq V2 | 0.5 | V2 证据 + 命中奖励**减半** |

- **V1 / V2**：低频证据表的两个版本（V2 为正式验证版，证据分更细）。
- **train-label-bonus**：选中 train driver 基因时的直接命中奖励，默认 1.0；0.0 = 完全去掉，0.5 = 减半。
- 统一参数：feature_mode=hybrid6_raw、max_episodes=50、seed=0、lowfreq bonus scale=1.2396 / cap=1.0。

### 三、结果（验证集指标，最佳 checkpoint）

| run | 配置 | NDCG@150 | Recall@150 | 命中数 | MRR | 平均排名 |
|---|---|---|---|---|---|---|
| **exp6** | **V2 + 去命中** | **0.0785** | **0.08** | **2** | **0.0143** | **2528** |
| exp4 | V1 + 减半 | 0.0700 | 0.08 | 2 | 0.0108 | 3394 |
| exp5 | V2 + 默认 | 0.0656 | 0.08 | 2 | 0.0086 | 5313 |
| exp3 | V1 + 去命中 | 0.0651 | 0.08 | 2 | 0.0087 | 4343 |
| exp1 | legacy 基准 | 0.0615 | 0.04 | 1 | 0.0139 | 3689 |
| exp7 | V2 + 减半 | 0.0615 | 0.04 | 1 | 0.0142 | 2603 |
| exp2 | V1 + 默认 | 0.0530 | 0.04 | 1 | 0.0109 | 1517 |

### 四、结论

1. **低频证据有效**：所有 lowfreq 配置的 Recall ≥ 基准（命中 1 → 2）。
2. **去掉 +1.0 命中奖励有帮助**：V2 下，去命中（exp6, 0.0785）> 默认（exp5, 0.0656）。
3. **最佳配置 = lowfreq V2 + train-label-bonus 0.0**（exp6），相比 legacy 基准 NDCG@150 提升约 28%，Recall 翻倍，driver 平均排名从 3689 → 2528。

### 五、局限与下一步

- **局限**：单 seed（seed 0）、绝对值仍弱（Recall 0.08 = 25 个 driver 只找到 2 个）。
- **下一步**：用 exp6 配置（V2 + 去命中）跑多种子（seed 42/43/44）验证稳定性；确认后可继续叠加 GRN SimpleUnion 或调 bonus scale。

---

## 多种子验证（2026-08-31）

### 一、目的

上一节低频实验在**单 seed（seed=0）**下得出「exp6（V2 + 去命中）最佳」的结论。为确认该结论是否稳定，本节用两个配置 × 三个随机种子做**配对对比**：

- **legacy 基准**（旧方法，train-label-bonus=1.0）；
- **exp6 最佳配置**（lowfreq V2 + train-label-bonus=0.0，即「V2 + 去命中」）。

配对方式：同一 seed 下比较 legacy vs V2+去命中，消除随机初始化差异，共 3 组配对。

### 二、计划（6 个 run，统一 50 轮、feature_mode=hybrid6_raw、lowfreq bonus scale=1.2396 / cap=1.0）

| run | 配置 | seed |
|---|---|---|
| exp1_legacy_seed42 | legacy | 42 |
| exp1_legacy_seed43 | legacy | 43 |
| exp1_legacy_seed44 | legacy | 44 |
| exp6_v2bonus0_seed42 | V2 + 去命中 | 42 |
| exp6_v2bonus0_seed43 | V2 + 去命中 | 43 |
| exp6_v2bonus0_seed44 | V2 + 去命中 | 44 |

- 命令：`bash scripts/run_multiseed_validation.sh`（脚本已提交）。
- 每 run 约 50 分钟，总耗时约 5 小时。
- 过程插曲：训练期间曾误启动第二份重复进程，在第 1 轮即发现并终止；其残留的半成品目录 `seed_42_20260831_143647` 已清理，不影响任何正式结果。

### 三、结果（验证集指标，best checkpoint，即 50 轮内 NDCG@150 最优的那一集）

| seed | 配置 | best轮 | NDCG@150 | Recall@150 | 命中数@150 | MRR | 平均排名 |
|---|---|---|---|---|---|---|---|
| 42 | legacy | 2 | 0.0615 | 0.04 | 1 | 0.0137 | 4088 |
| 42 | V2+去命中 | 5 | 0.0615 | 0.04 | 1 | 0.0138 | 3746 |
| 43 | legacy | 8 | 0.1256 | 0.20 | 5 | 0.0105 | 4353 |
| 43 | V2+去命中 | 1 | 0.1087 | 0.20 | 5 | 0.0061 | 2304 |
| 44 | legacy | 15 | 0.0792 | 0.08 | 2 | 0.0141 | 3125 |
| 44 | V2+去命中 | 14 | 0.0703 | 0.08 | 2 | 0.0107 | 3206 |
| **平均** | **legacy** | — | **0.0888** | **0.107** | **2.67** | **0.0128** | **3856** |
| **平均** | **V2+去命中** | — | **0.0802** | **0.107** | **2.67** | **0.0102** | **3085** |

（对比图见 `outputs/multiseed_compare.html`，含训练曲线、验证曲线、箱线图与基因排名图。）

### 四、结论

1. **单 seed 的「exp6 最佳」结论未被确认。** 3 个种子下 V2+去命中没有稳定胜过 legacy：NDCG@150 平均反而略低（0.0802 vs 0.0888），命中数与 Recall 完全打平，MRR 略低。此前 seed=0 的领先属于**随机种子碰巧较好**，不是配置的真实优势。
2. **两个假设均未通过多种子验证。**
   - 假设 A（低频证据有效）：3 个种子下命中数逐一打平（1/1、5/5、2/2），低频证据未带来更多命中；
   - 假设 B（去掉 +1.0 命中奖励有助泛化）：NDCG 平均反而 legacy 略好。
3. **种子间方差远大于配置差异。** 同配置换 seed，命中数可在 1↔5 间大幅波动；而同一 seed 换配置，结果几乎不变。说明当前评估主要反映「该 seed 的运气 + 静态结构」，而非配置效果。
4. **奖励对最终排名的支配力很弱。** 同 seed 42 下，legacy 与 V2+去命中两个 run 学出的排名几乎一致：top 基因同为 TP53/EP300/CREBBP，命中的 driver 及名次完全相同（NDCG@150 精确到 17 位小数都相等），仅 GRB2/ESR1 等个别位置微调，且 V2 的 Q 值整体约高 1.4 倍。这说明换奖励配方基本不改变模型学到的排序模式——**排名主要由静态特征/图结构决定，而非强化学习信号**。
5. **唯一正向线索**：V2+去命中的平均排名（25 个 driver 的平均位次）更好（3085 vs 3856），说明它把 driver 整体排得略靠前，但幅度不足以挤进 top-150。

### 五、局限与下一步

- **局限**：仅 3 个种子、50 轮、单数据集；best checkpoint 都取得很早（第 1~15 轮），后半程 NDCG 波动大、无明显收敛，反映训练不稳定。
- **对上一节结论的修正**：上一节「最佳配置 = exp6（V2 + 去命中），NDCG@150 提升约 28%、Recall 翻倍」仅在 seed=0 下成立，多种子下不成立。
- **下一步方向（不再沿奖励配方微调）**：
  1. **贪心 rollout 顺序选择**（已记录于 AGENT_HANDOFF_REPORT §11.6）：当前评估是单遍「空上下文」打分排序，没用上模型的序列决策能力；改成「逐个选基因、已选基因不再重选」的顺序排序，可能更贴近 RL 本质。
  2. **特征/图判别力验证**：先确认静态特征本身能否把 driver 与普通基因区分开（如特征分布对比、简单监督基线），再谈 RL 改进。
  3. 若继续 RL 路线，优先解决**训练不稳定**（学习率、探索率、评估频率等）。

---

## RL 学习回路审计与修复（2026-09-01）

本轮先停掉一切新方向（MORL / GRN / 新癌种 / 新低频奖励版本），只做两件事：
① 审计并修复 RL 学习回路本身；② 用 legacy/recovery 目标验证修复是否有效。
审计发现：**RL 学不进去的主要责任不在「奖励配方」，而在训练机制和目标错位。**
以下是本轮确认的错误原因（按严重程度排列）：

### 一、五个已确认的错误原因

1. **reward 在 `learn()` 里又除以了 10，reward 在 TD 目标中占比极低。**
   奖励本身已经很小（legacy 每步均值 ≈ 0.012），`learn()` 里又执行 `reward_sum = reward_sum / 10.0`，
   使每步 reward 变成 ≈ 0.0012，而 `gamma × Q_next` 约在 1~5 的量级——**reward 对 TD target 的贡献 < 1%**，
   目标函数退化为「让 Q 自洽地逼近 gamma×Q_next」的价值平滑，奖励信号被完全淹没，
   agent 实际上不在学奖励，只是在自说自话。已删除该 `/10.0`（`src/DQN.py`）。

2. **PER（优先级经验回放）参数设置使其近似均匀采样。**
   `--per_alpha 0.2` 接近均匀；`--per_beta_frames 2,000,000` 远大于整个训练的学习步数（50 轮 × 150 步 ≈ 7,500），
   beta 全程卡在 ≈0.1 不升——重要性采样权重形同虚设，高 TD 误差样本不被聚焦。已改为
   `alpha=0.6`、`beta_frames` 自动取 `max_episodes × topk`（训练结束时 beta 恰好升到 1.0）（`src/train.py`）。

3. **梯度长期被裁剪到 1.0。**
   `--gradient_clip 1.0` 对约 16 万参数的网络（smooth_l1 损失下梯度范数天然在 13~236）而言，
   是**每一步都满强度裁剪**，有效步长被压死到 `lr × 1.0`，网络基本不动。已改为 50.0（`src/train.py`）。

4. **V2 奖励与突变频率强负相关，而三套 driver 标签全部集中在突变频率高百分位。**
   Train(16)/Validation(25)/Test(18) 的突变频率百分位中位数分别为 **99.8% / 98.5% / 95.1%**
   （相对 9039 个背景基因），VHL(0.462) 是全局最高突变基因；
   而低频证据分 V2 与突变频率的 Spearman ≈ **−0.59**，三套标签的证据分均值只有 0.16~0.28（背景 0.50）。
   **奖励在找「低突变新候选」，标签在评「高突变已知 driver」，两者方向相反。**

5. **因此出现「模型 reward 上升、Validation NDCG 下降」——本质是「训练目标与评价目标冲突」。**
   审计中 V2 的 episode_reward 确实在上升（agent 学到了东西），但 val_ndcg 塌到 0.00~0.05：
   agent 在高效地优化一个与评价反着来的目标。这不是调 reward 配方能解决的，是任务定义与评估口径的问题。
   —— 副产品：由于已知 driver 就是高突变基因，**单按突变频率静态排序就能命中 13/25 验证 driver**，
   这是标签偏置的镜像，不是生物学信号；也是当前所有学习路线（RL / 监督）都难以企及的墙。

### 二、已做的修复（只动已确认问题，不新增模块）

| 项 | 修复前 | 修复后 | 位置 |
|---|---|---|---|
| reward 缩放 | `learn()` 内再 `/10.0` | 不再缩放 | `src/DQN.py` |
| PER alpha | 0.2（≈均匀） | 0.6 | `src/train.py` |
| PER beta 周期 | 2,000,000（beta 卡 0.1） | 自动 = `max_episodes×topk` | `src/train.py` |
| 梯度裁剪 | 1.0（全程强裁） | 50.0 | `src/train.py` |
| epsilon 下限 | 0.15（每轮约 22 个随机动作） | 0.05 | `src/train.py` |

### 三、修复验证（训练前 vs 训练后）

- 训练：`bash scripts/run_mechanism_fix_validation.sh`（legacy × seed 42/43/44，
  新默认机制参数，输出到 `outputs/fix_legacy_seed*/`；**正式训练请用户手动运行**）。
- 评估：脚本会接着跑 `evaluate_greedy_rollout.py`（NDCG@150 / Recall@150 / HitCount@150）
  与 `scripts/compare_prepost_fix.py`（Top-150 重合 + Spearman 秩相关）。
- 判读口径（写死在 `compare_prepost_fix.py` 里）：
  - 修复有效：NDCG/Recall 明显上升，同时 top-150 重合率下降、rank corr 明显 < 1 → 训练真的在改排序，且方向与评价一致；
  - 修复无效：前后几乎不动、top-150 重合≈150、rank corr≈1 → 学到 = 没学；
  - 病态：NDCG 下降但 reward 上升 → 训练目标仍与评价目标冲突（回到本审计第 4、5 条）。
- 对照基准（修复前的旧 legacy 三种子，`outputs/rollout_eval/prepost_OLD_legacy_baseline.csv`）：
  训练前后 NDCG 涨跌靠运气（+0.048→0.061 / +0.056→0.126 / −0.104→0.079），
  seed43 排名被彻底打乱（Spearman −0.616）、seed44 反而下降——训练不稳定、且对评价的提升不可复现。

### 四、下一步决策门

- 修复后 legacy 训练若**明显改善**（见上）→ 恢复 Recovery–Discovery 权重扫描，回到 MORL 主线；
- 若仍**无明显改善** → 暂停 MORL。考虑到第 1 条（静态突变排序即 13 命中的墙）与第 4、5 条（目标错位），
  更可能的结论是需要**重新定义任务**：要么接受「恢复已知 driver」以静态基线为准，要么为「发现低频新候选」
  另立评价协议（低频标签或外部验证），而不是继续调 RL。

### 五、修复验证结果（2026-09-02，正式训练 3 种子已完成）

机制修复验证 **2/3 种子明显有效**。训练前 → 训练后（best checkpoint，一次性/rollout 取较高）：

| 种子 | NDCG@150 前→后 | Recall 前→后 | Hits 前→后 | top150 重合 | Spearman |
|---|---|---|---|---|---|
| 42 | 0.048 → **0.171/0.185** | 0.04 → **0.32/0.36** | 1 → **8/9** | 32/150 | −0.37 |
| 43 | 0.056 → **0.188/0.140** | 0.12 → **0.36/0.28** | 3 → **9/7** | 91/150 | +0.29 |
| 44 | 0.104 → **0.083/0.057** | 0.20 → 0.12 | 5 → 3 | 22/150 | −0.68 |
| 均值 | → **0.147** | → **0.27** | → **7.0** | 48/150 | −0.25 |

**结论一：修复有效，机制问题确实是主因。** 对比修复前旧 legacy（同 seed）：平均命中从 **3.0 → 7.0**、
平均 NDCG 从 **0.090 → 0.147**；seed42/43 各命中 9 个验证 driver、NDCG 0.17~0.19，**打破 RL 历史纪录（旧最好 7 个/0.139）**，
并且**已反超同架构监督 GCN（7~8 个）**。训练前后排序被实质改写（Spearman 远离 1、top150 重合平均仅 48/150），
证明去掉 reward/10 + PER/裁剪/epsilon 修复后，reward 信号真正开始驱动学习。

**结论二：仍有三道硬伤，未过"超过静态基线"这一关。**
1. **仍落后静态突变基线**：最好命中 9 < 静态突变 13（NDCG 0.188 < 0.283）——标签偏置的墙依然在；
2. **种子不稳定**：seed44 训练后反而倒退（5→3 命中），种子间方差依旧很大；
3. **best checkpoint 都落在第 1~3 轮，之后 NDCG 衰减到 0.04~0.06**，而 reward 持续走高（~27 不再涨但 NDCG 不跟）——
   证明 reward 与评价目标**部分对齐但未完全**：早期训练（前几轮）学对了方向，继续训练后 reward 驱动又把它拉歪，
   是审计第 4、5 条（目标冲突）在更小尺度上的残留。这也解释了为什么"每轮验证、取 best"才能保住好结果。

**下一步选项（待用户定夺，见对话）**：
- 选项 A：按本轮指令"明显改善"达标 → 恢复 **Recovery–Discovery 权重扫描**，测试能否稳定种子 + 抵消目标冲突；
- 选项 B：考虑到"仍未超过静态突变基线 + 种子不稳定 + best 只在前 3 轮"这三条 → **重新定义任务**优先。
- 选项 C：先修 reward-评价对齐（如把 train_label_bonus 做成主要信号 / 显式奖励"排进 top-k 的已知 driver"），再回头扫描。

## 修复第二阶段：多 seed 稳定性验证（2026-09-02）

用户决定（约束条件）：
- 当前修复机制**原样不动**，补跑 seed 45-49（机制参数与 42/43/44 完全一致）；
- 不改 reward 内容、不做 MORL / GRN / 新癌种；
- 只允许调 lr / epsilon 衰减 / 更新频率等稳定参数（**禁止**新增「Train driver 进 Top-K」奖励，避免进一步过拟合 16 个训练标签）；
- 不要求 RL 必须超过突变频率 baseline（突变只作 Recovery 强基线），最终目标仍是 Recovery–Discovery trade-off；
- 若 5 seed 下修复可重复且 reward 稳定改变排序 → 进 Recovery–Discovery 权重扫描；否则继续定位训练稳定性/泛化。

**8 seed（42–49）验证已完成**（`outputs/fix_all_rollout_eval/`：stability_curves.csv / pre_best_last_compare.csv；
汇总脚本 `scripts/analyze_stability_fix.py`）。机制与第一阶段完全一致，仅换 seed。

每 seed 曲线证据（`train_metrics.csv`）：

| seed | 峰值轮 | best NDCG | 末轮 NDCG | 峰值后跌 | reward 持续走高 | 说明 |
|---|---|---|---|---|---|---|
| 42 | 3 | 0.171 | 0.044 | 74% | ✓ | 前3轮见顶 |
| 43 | 1 | 0.188 | 0.053 | 72% | ✓ | 前3轮见顶 |
| 44 | 2 | 0.083 | 0.048 | 42% | ✓ | 前3轮见顶 |
| 45 | 1 | 0.182 | 0.053 | 71% | ✓ | 前3轮见顶 |
| 46 | 4 | **0.211** | 0.048 | 78% | ✓ | 见顶略晚，命中10 历史最高 |
| 47 | 43 | 0.072 | 0.056 | 22% | ✓ | **例外：从未冲高（dud）** |
| 48 | 3 | 0.191 | 0.044 | 77% | ✓ | 前3轮见顶 |
| 49 | 2 | 0.131 | 0.053 | 60% | ✓ | 前3轮见顶 |
| 均值 | 中位 2.5 | 0.154 | 0.050 | −62% | 8/8 | — |

三时点一次性评估（8 seed 平均）：NDCG pre 0.058 → **best 0.154** → last 0.050；
Recall 0.08 → **0.28** → 0.045；命中 pre 2.0 → **best 7.0** → last 1.1；
best↔last 的 top150 重合平均 46/150、Spearman −0.22 —— **best 和 last 基本是两个不相干的模型**。

**结论（8 seed 正式判定）**：
1. **「前 1-3 轮见顶、reward 走高而 NDCG 掉」是稳定主模式**：6/8 严格前 3 轮见顶，加 seed46（第 4 轮）
   = 7/8 极早期见顶；唯一例外是 seed47（dud，全程未冲高，~1/8 dud 率）。继续训练会毁掉早期好排序，
   提升只活在 best checkpoint —— 训练目标与评价目标部分冲突（审计第 5 条残留）在 8 seed 下稳定复现，
   **过拟合 16 个训练标签（train_label_bonus）是最可能病灶**。
2. **修复可重复性达标**：训练前 → best NDCG 平均 **+0.096**（8 seed 中 7 个为正）、命中 2.0→7.0、
   reward 8/8 稳定改写排序 → **按用户门槛可进入 Recovery–Discovery 权重扫描**。
3. **仍落后静态突变基线**（best 命中 7.0 < 突变 13、NDCG 0.154 < 0.283）：突变只作 Recovery 基线，
   目标仍是 Recovery–Discovery trade-off，不要求必超。
4. **风险提示**：best 状态只活在前 1-4 轮；每配置应多 seed 取稳健结论以对冲 1/8 dud；
   train_label_bonus 旋钮恰在权重扫描调节范围内，正式扫描前值得先小试降权以确认崩塌元凶。

**状态（2026-09-02）**：已整理 8 seed 数据供用户 review（`outputs/fix_all_rollout_eval/`），
下一步（权重扫描 / 稳定性探针）待用户定夺。

---

## Recovery–Discovery 权重探针（2026-09-02，代码就绪，待正式运行）

用户最终裁定（最小方案）：固定当前已修复 RL 机制、**不再改 reward 结构**；
保留 legacy recovery 信号 + 启用 evidenceV2 discovery 信号，新增
`reward = w_recovery × recovery + w_discovery × discovery`，
测 **(1,0)/(0.8,0.2)/(0.5,0.5)/(0.2,0.8)/(0,1)** 五组权重 × **3 seed** × **10 轮**。

**Discovery 目标先定义为「低频新候选」，不把全部低频基因算 discovery**：
- 低频区 = 突变患者数 **2–18**；
- **LowFreqNovel@150** = top-150 ∩（低频区 且 ∉ train∪val 已知 driver）→ 共 **2419** 个；
- **EvidenceSupportedLowFreqNovel@150** = 上述且 `LowFrequencyEvidenceScoreV2 ≥ 0.5`（低频 novel 的顶部 ~10%）→ 共 **224** 个；
- reward 侧 discovery 门控（frozen-label，不读 val）：低频区 + evidence≥0.5 + 非 train driver。

共享定义在 `src/rd_definitions.py`（reward 侧与评估侧同源）；判定标准：五组权重下
Recovery 指标（NDCG@150/Recall@150）随 w_discovery 升高而**一致下降**、
Discovery 两指标**一致上升**（3 seed 符号一致）→ 进 preference-conditioned MORL，否则暂停重新定义。

**实现 + 冒烟已通过**（未污染正式产物）：
- `src/DQN.py` 新增 `rd_scan` 模式；`src/train.py` 新增 `--w-recovery/--w-discovery/--rd-evidence-*`；
- `scripts/evaluate_greedy_rollout.py` 新增 LowFreqNovel@150 / EvidenceSupportedLowFreqNovel@150 两列；
- 冒烟：rd_scan(1,0) 与 legacy 逐轮 reward/NDCG **完全一致（差值 0）**；rd_scan(0,1) 只从 supported
  低频新候选拿分、train/val driver 零污染；候选集计数 2419/224 与设计一致。

**正式训练由用户手动运行**（GPU 约 3–4 小时，建议后台/过夜）：
```bash
nohup bash scripts/run_rd_probe.sh > outputs/rdprobe.log 2>&1 &
```
产物：`outputs/rdprobe_r*_seed{42..44}/`（15 个 run，每轮 train_metrics.csv）、
`outputs/rdprobe_eval/`（summary_metrics.csv 含四指标、rankings/、
`rd_probe_group_summary.csv` mean±SD、`rd_probe_direction.csv` 方向性 + 判定）。

---

## Recovery–Discovery 权重探针：rollout 主评价重评（2026-09-03）

### 一、评价口径变更及原因

15 个既有 run 均已完成，本轮**没有重新训练**其中任何一个。此前的主结果以空上下文的 one-pass Q
ranking 选择 checkpoint；但 one-pass 的 top-150 与序列 greedy rollout 经常不一致，前者不能充分评价
RL 的序列决策策略。因此从本轮起：

- **greedy rollout 是 RL 的主评价与 checkpoint 选择口径**；one-pass Q ranking 仅保留为辅助诊断；
- Recovery：`NDCG@150`、`Recall@150`；
- Discovery：`HighEvidenceLowFreqNovel@150`、`DiscoveryPrecision@150`
  （高证据低频新候选 / 低频新候选）和 `DiscoveryFoldEnrichment@150`
  （该 precision 相对候选池内高证据比例的倍数）；
- 静态 mutation、degree、evidenceV2 基线用相同 Discovery 指标比较。

低频新候选池为 2,419 个，其中高证据候选 224 个，故候选池高证据基线比例为
`224 / 2419 = 9.26%`，fold enrichment 为 1 表示与候选池随机组成相同。

**既有 checkpoint 限制**：旧训练只保存了 `checkpoint_best.pt`（按旧 one-pass 口径选出）与
`checkpoint_last.pt`，没有逐 episode checkpoint。因而本次只能在这两份**已保留 checkpoint**中按 rollout
Recovery NDCG 选择主 checkpoint（15 个 run 中 6 个选 best、9 个选 last），不能把它表述为完整的逐轮
rollout early stopping。后续新训练应逐轮保存 checkpoint，并直接按 rollout 选择。

### 二、rollout 主口径结果（3 seed 均值±SD）

| (w_recovery, w_discovery) | NDCG@150 | Recall@150 | 高证据低频新候选 | Discovery Precision | Fold enrichment |
|---|---:|---:|---:|---:|---:|
| (1.0, 0.0) | 0.2231±0.0126 | 0.4267±0.0231 | 6.00±2.00 | 5.40%±2.13% | 0.58±0.23 |
| (0.8, 0.2) | 0.2010±0.0571 | 0.3867±0.1007 | 5.00±1.00 | 4.42%±0.97% | 0.48±0.10 |
| (0.5, 0.5) | 0.1853±0.0686 | 0.3467±0.1155 | 6.67±4.04 | 6.62%±4.84% | 0.71±0.52 |
| (0.2, 0.8) | 0.1397±0.0235 | 0.2533±0.0231 | 5.67±3.06 | 6.18%±4.16% | 0.67±0.45 |
| (0.0, 1.0) | 0.0667±0.0237 | 0.0800±0.0693 | 10.00±2.65 | 19.44%±4.05% | 2.10±0.44 |

方向性检验采用每个 seed 内 5 个权重的 Spearman 相关：

- Recovery 随 `w_discovery` 上升稳定下降：NDCG 总体 ρ=−0.775、每 seed 为
  −0.90/−0.90/−0.60；Recall 总体 ρ=−0.830、每 seed 为 −0.97/−0.97/−0.67；
- Discovery Precision / Fold enrichment 稳定上升：两者总体 ρ=+0.551、每 seed
  为 +0.80/+0.15/+0.80；
- 高证据低频新候选数从端点均值 6.0 增至 10.0（总体 ρ=+0.374，三 seed 均为正）；
- 宽泛的低频新候选总数从 113 降至 53，这是较高 discovery 权重筛向高证据子集的预期组成变化，
  不再作为 Discovery 主优化指标。

因此，按照本轮预注册的 rollout 主口径，**3 个 seed 的 Recovery–Discovery 方向性一致，探针达到了进入
preference-conditioned MORL 的资格条件**；本轮本身没有启动 MORL 训练。

### 三、静态基线与边界

| 方法 | NDCG@150 | Recall@150 | 高证据低频新候选 | Discovery Precision | Fold enrichment |
|---|---:|---:|---:|---:|---:|
| static mutation | 0.2826 | 0.5200 | 0 | 0.00% | 0.00 |
| static degree | 0.0530 | 0.0400 | 10 | 23.81% | 2.57 |
| static evidenceV2 | 0.0000 | 0.0000 | 0 | 0.00% | 0.00 |

纯 discovery 的 rollout 配置 `(0,1)` 已把 Discovery precision 提升到 19.44%、fold enrichment 2.10，
但仍低于 static degree 的 23.81% / 2.57；同时 Recovery 明显损失。该基线差距必须在后续 MORL 设计和
报告中保留，不能只报告 RL 的相对改善。

### 四、产物与标签隔离

- 新重评目录：`outputs/rdprobe_rollout_primary_eval/`；
  `summary_metrics.csv` 为逐 run 指标，`rd_probe_group_summary.csv` 为组汇总，
  `rd_probe_direction.csv` 为方向性检验，`rd_probe_static_baselines.csv` 为静态基线；
- 评价脚本中的 Discovery 已知集合只使用 Train∪Validation；Recovery 只读取 Validation 标签；
  **Test 标签未读取、未参与 checkpoint 选择、权重选择或任何本轮结论。**

---

## Preference-conditioned MORL 实现与公平验证（2026-09-03）

在 rollout 权重探针通过方向性门槛后，项目进入共享 MORL 实现阶段。实现严格保持既有
`rd_scan` Recovery / Discovery reward 定义和参数不变；改变的仅是 Q 网络额外接收
`w=(w_recovery,w_discovery)`，并让同一模型学习多个 preference。

- 训练 preference（已见）：`(1,0)/(0.8,0.2)/(0.5,0.5)/(0.2,0.8)/(0,1)`；每轮从这些
  preference 平衡采样，replay transition 同时存储对应 preference；
- 评估 preference（未见插值诊断）：`(0.9,0.1)/(0.65,0.35)/(0.35,0.65)/(0.1,0.9)`；
  未见 preference 不参与训练或 Pareto checkpoint 选择；它只检验连续 preference 输入的插值，
  **不构成生物学或最终泛化证据**；
- 每轮主评价均为 greedy rollout，记录 Recovery NDCG@150 / Recall@150 与 Discovery
  Precision / Fold Enrichment；one-pass 不参与模型选择；
- checkpoint 以每个已见 preference 的 rollout 三维向量
  `(NDCG@150, Recall@150, DiscoveryPrecision@150)` 求 Pareto non-dominated 前沿，保留所有
  任一已见 preference 前沿上的 checkpoint；Fold Enrichment 与 Precision 在固定候选池中完全共线，
  因此记录但不重复放入支配判定；
- 每个 MORL run 的 retained Pareto 点会与相同 seed / preference 的 15 个 scalarized RL run 直接比较，
  报告共享模型是否覆盖或接近已有 frontier；
- 2026-09-03 的首次 3-seed × 10 episode 运行只为实现冒烟：共享模型对每个 preference
  平均只收集 2 个 episode，但每个 scalarized 对照各有 10 个 episode。因此它不能用于否定
  MORL，也不作为是否扩至 5 seed 的决策依据；其历史比较结果保留在
  `outputs/morl_shared_10ep_comparison/morl_vs_scalar_frontier_coverage.csv`；
- 正式公平验证固定为 3 seed × 50 episode。五个已见 preference 使用同一随机五项循环，
  各**恰好**训练 10 个 episode；训练摘要保存 `trained_preference_counts`，启动脚本会验证
  每项均为 10 后才进入比较；
- 正式验证顺序：只有 50-episode 结果中的 trade-off、未见 preference 插值和 scalar frontier
  对比在 3 seed 稳定，才扩至不少于 5 seed；**Test 始终不参与训练、checkpoint、preference
  或模型选择。**

实现文件：

- `scripts/train_preference_morl.py`：共享 preference Q、条件 replay、rollout Pareto checkpoint；
- `scripts/analyze_preference_morl.py`：对 scalarized frontier 的逐 seed 覆盖比较；
- `scripts/audit_preference_conditioning.py`：审计 `w` 的 Q→PER→sample→online/target TD 链，
  并可对已训练 checkpoint 做固定 state 条件响应测试；
- `scripts/run_preference_morl_validation.sh`：双头 vector-Q MORL 的 50-episode 公平 3-seed 启动脚本；为每次运行创建
  唯一 output group，并在比较 CSV 未成功写入时失败退出。

### 当前双头 vector-Q MORL 修复（尚未产生新的正式结果）

针对 scalarized MORL 的 scale dominance 与 shared-trunk gradient conflict，当前实现改为：

- Q 网络输出固定的两个头 `[Q_recovery, Q_discovery]`；两个头**不再接收** preference，避免
  head 本身随 `w` 改变而失去可分解含义；
- replay 保存 reward 向量，两个目标独立建立 Double-DQN TD target 和 Huber loss；Recovery 与
  Discovery reward 分别是既有 `rd_scan` 在 `(1,0)`、`(0,1)` 的未加权端点值，仍按原来的 `[0,5]`
  边界裁剪，不修改 reward 配方；
- 仅在动作选择时，以每头 TD-target 绝对值 EMA 作尺度归一化后合成
  `w_recovery * Q_recovery + w_discovery * Q_discovery`。该 EMA 同时用于每头 loss 和 PER priority，
  防止绝对数值更大的头支配学习；
- checkpoint 保存该尺度，保证 greedy rollout 与后验诊断使用训练时相同的合成口径；Test 仍完全冻结。

2026-09-04 的 target/bootstrap 审计确认 terminal mask 正确，但发现非终止 TD bootstrap 可比即时
reward 大数十至数千倍，且 seed45 的 target 明显大于已记录轨迹的折扣 return。因而新的 vector-Q
训练默认使用 **PopArt**：每头网络预测标准化 return；EMA mean/std 更新时同步重标定 online 与 target
的输出层，保持原始 Q 不突变。它取代旧版“只在 loss 上除以 scale”的做法，不改 reward、gamma、
replay、greedy rollout 或 Test 隔离。旧 vector-Q checkpoint 明确标为 `legacy_loss_scale`，防止混用。

`scripts/run_preference_morl_validation.sh` 现启动 seed 42/45/46 的 50-episode PopArt vector-Q 验证，
每个已见 preference 恰好收集 10 个 episode；完成后自动进行同 seed scalar 对照、policy interpolation、
head-scale 与 Recovery/Discovery objective-gradient cosine 的只读诊断。新诊断由
`scripts/diagnose_vector_morl.py` 产生；它明确不把未序列化的历史 PER 采样/priority 伪装为可观测值。
每个 episode 的两头尺度及优化摘要另写入该 run 的 `morl_training_metrics.csv`，便于检查尺度主导是否
在训练中形成。

该诊断还审计终止 transition：每个被抽样的终止项必须满足 `TD target == immediate reward`；非终止项
单独记录 bootstrap 项相对即时 reward 的量级。它用于区分 terminal mask 错误与非终止 Q 自举增长，
不参与训练或 checkpoint 选择。

自动条件审计已通过：同一批 preference 完整进入两次 online-Q 与一次 target-Q，且
preference encoder 梯度非零；对已训练的 10-episode checkpoint，固定 state 下不同 preference
产生不同 Q 值与部分 Top-150 排序差异。这证明条件链未断，但不证明其偏好 trade-off 已学习成功。

---

## 主线切换：preference-conditioned contextual bandit（2026-09-08，代码与冒烟完成，正式训练待运行）

基于 RL 必要性验证与 DDQN 历史状态消融，当前主线暂时从序列 DDQN / MORL 切换为共享的
**preference-conditioned contextual bandit**。本轮的目的仅是检验：在不使用
`gamma * Q_next` bootstrap 的情况下，一个共享模型能否根据 preference 学到稳定的
Recovery–Discovery 排序取舍。

- 保持 `hybrid6_raw` 特征（Degree、WeightValue、PatientCoverageCount、Mutation、Expression、
  Methylation）和 PPI 图不变；当前正式链路**没有启用 GRN**，因此本轮不会重新加入 GRN；
- 保持冻结的 `rd_scan` Recovery / Discovery 定义、evidenceV2 表、Train/Validation 划分不变；
- Q 网络、PER transition、PER sample、online Q 和即时 TD loss 均接收同一个
  `w=(w_recovery,w_discovery)`；bandit 的目标严格为 `Q(s,a,w) = r(s,a,w)`，不读取或使用
  `Q_target(s',a')` 建立 TD target；
- 每个 seed 用一个共享模型训练 50 episode，五个已见 preference
  `(1,0)/(0.8,0.2)/(0.5,0.5)/(0.2,0.8)/(0,1)` 各恰好 10 episode；最终仅评估唯一的
  final checkpoint 的 greedy rollout，不以 Validation 指标挑选 checkpoint；
- 正式比较固定为 seed 42/45/46，输出 NDCG@150、Recall@150、Discovery Precision、
  Fold Enrichment 与跨 seed Top-150 Jaccard，并和既有 scalar DDQN、scalar bandit、
  static mutation、MLP、GCN 同表报告；Test 标签不读取、不参与任何训练或决定。

实现文件：

- `scripts/train_preference_bandit.py`：共享条件 bandit 训练与 final greedy rollout；
- `scripts/run_preference_contextual_bandit.sh`：3 seed 正式启动与汇总；
- `scripts/analyze_preference_bandit.py`：统一输出逐 seed、mean±SD、Jaccard、paired difference
  和每 seed 的 preference 方向性结果。

**公平性边界**：与每个 scalar 对照相比，每个 preference 都有相同的 10-episode 直接经验；但共享模型
还会从另外四个 preference 的 40 episode 获得共享表示更新。因此该比较适合回答“共享条件模型是否能以
每个条件 10 episode 覆盖多个偏好”，**不**能表述为总优化计算量完全相同的因果比较。报告中必须同时保留
这一限制以及 3 seed 仅能支持描述性方向检验、不能声称统计显著性的限制。

### 首次 3-seed 运行审查（2026-09-08）：不进入 5-seed 扩展

首次输出 `outputs/preference_bandit_20260908_152823_comparison/` 的三个 run 均完整（50 episode，
每个 preference 10 次），模型也确实是即时 reward bandit，Test 未读取。聚合均值表面上呈现
Recovery 从 NDCG `0.1657` 降至 `0.1189`、Discovery Precision 从 `14.54%` 升至 `18.85%` 的端点
变化，且 Top-150 Jaccard 在各 preference 为 `0.51–0.55`，高于对应 DDQN 的 `0.08–0.19`。

但该结果**不能**据此扩到 5 seed：seed42、46 的方向正确，seed45 却反转
（Recovery NDCG vs `w_discovery` Spearman `+0.40`，Discovery Precision `-0.95`）。更关键的是，
首次实现把每 seed 的一个随机五项排列重复十次，使每个 preference 永远处于相同的 block 位置；
preference 因而与 epsilon、replay 年龄和参数训练时间混杂。固定-state 审计显示三个模型都对
preference 有 Q 值和排序响应，所以它不是条件输入断开的证据，而是**调度混杂下无法判定的
训练稳定性结果**。

正式重跑改用 `balanced_latin_blocks_v1`：每 5 个 block 构成 Latin square，每个 preference 在五个
block 内的每个位置各一次；50 episode 下每项在每个位置各出现两次，仍保持总 exposure=10。只重跑
seed42/45/46；不改变 reward、特征、网络、PER、评价、基线或 Test 隔离。旧结果保留为诊断记录，
详见 `outputs/preference_bandit_20260908_152823_comparison/REVIEW.md`。

### Phase-balanced 3-seed 验证（2026-09-08）：通过扩展门槛

修正调度后的 `outputs/preference_bandit_20260908_171404_comparison/` 已完成。三个 seed 均满足
每个 preference 10 次、每个 block 位置各 2 次；即时 bandit target、greedy final rollout 与 Test 冻结
也均经配置审计确认。

| 指标 | `(1,0)` | `(0,1)` | 3-seed 方向性 |
|---|---:|---:|---|
| NDCG@150 | 0.1479±0.0599 | 0.0885±0.0713 | seed42/45/46: `−1.00/−0.70/−0.70` |
| Recall@150 | 0.2667±0.1222 | 0.1733±0.1405 | `−0.95/−0.87/−0.82` |
| Discovery Precision | 21.39%±8.47% | 24.43%±8.86% | `+0.90/+0.89/+0.97` |
| Fold Enrichment | 2.31±0.91 | 2.64±0.96 | `+0.90/+0.89/+0.97` |

**判定**：Recovery↓ / Discovery↑ 在 3 seed 均同向，且 shared bandit 的 Top-150 Jaccard 为
`0.26–0.36`，逐 preference 高于 scalar DDQN 的 `0.08–0.19`，因此满足预定的“扩至 5 seed”条件。

边界同样保留：它在 Recovery 端低于 scalar contextual bandit（NDCG 0.2224）和 static mutation
（0.2826），并且其 Jaccard 也低于 scalar contextual bandit；当前只能称为“相对 DDQN 更稳定地形成
trade-off”，不能称为最优模型或全面胜出。

下一步是 **5-seed shared-bandit 可重复性扩展**：保留已验证的 seed42/45/46，新增预先固定的
seed47/48。为遵守“暂停 DDQN/MORL”的范围，不重训不匹配的 DDQN、MLP、GCN；因此 5-seed 阶段只检验
shared bandit 本身的方向性与稳定性，跨方法的公平比较仍严格限于已有匹配的 3 seed。启动脚本为
`scripts/run_preference_contextual_bandit_5seed_extension.sh`。

### 冻结 checkpoint 的未见 preference 插值审计（2026-09-08）

使用 `scripts/audit_preference_bandit_interpolation.py` 对 5 个 final checkpoint 做只读 greedy rollout，
评估未见权重 `(0.9,0.1)/(0.65,0.35)/(0.35,0.65)/(0.1,0.9)`；没有继续训练、模型选择或 Test 读取。

结论为**部分通过**：5 seed 在 seen+unseen 九个权重上的 Recovery NDCG 相关均为负，Discovery
Precision 相关均为正，说明总体 trade-off 方向能够延伸到未见权重。但未见点只有 `11/20` 的 NDCG
落入相邻 seen 指标区间（Recall `18/20`、Discovery `17/20`）；seed48 在中间权重出现 Top-150
Jaccard `0.376/0.266` 的明显跳变。每 seed 的九个输入形成 5–9 个不同 Top-150 集合，故不是完整
conditioning collapse，但也不能声称连续平滑的 preference 泛化已经成立。

完整产物在 `outputs/preference_bandit_20260908_190650_interpolation/`。若必须支持任意连续权重，下一候选
是双即时价值 head 的 contextual bandit（分别回归 Recovery/Discovery reward，选择动作时再按 `w`
线性合成）；这仍不使用 bootstrap、不改变 reward，但属于新的结构验证，本轮尚未实施。

---

## 双即时价值 head contextual bandit（2026-09-08，代码与冒烟完成）

当前只验证 dual-head bandit，不继续修改单-head bandit、DDQN/MORL、reward 或数据。严格实现为：

- 共享原有 GCN/trunk，最后两个独立输出坐标分别预测
  `Q_recovery(s,a)` 与 `Q_discovery(s,a)`；它们的 TD target 分别是已有 `rd_scan` 的两个未加权即时
  reward，均不包含 `gamma * Q_next`；
- preference 不进入两个 head；动作选择严格使用原始价值
  `w_recovery * Q_recovery + w_discovery * Q_discovery`。每头 EMA scale 只平衡 Huber loss，不进入推理；
- target 网络在 contextual-bandit 模式下不前向、也不更新；单元审计确认 target forward=0、参数变化=0；
- 正式小规模验证固定 seed42/45/48、50 episode、phase-balanced 五个训练权重，并在 final checkpoint
  上同时评价五个已见权重和四个未见权重；Test 完全冻结。

实现与启动文件：

- `scripts/train_dual_head_bandit.py`：双 head 即时 reward 训练与 seen/unseen final rollout；
- `scripts/analyze_dual_head_bandit.py`：插值、Top-150 Jaccard、与 single-head shared/scalar bandit 的差值；
- `scripts/run_dual_head_bandit_validation.sh`：seed42/45/48 正式验证入口。

比较边界：single-head shared bandit 在 42/45/48 三个 seed 均有匹配结果；已有独立 scalar contextual
bandit 只有 seed42/45/46，因此双 head 对 scalar bandit 只能公平配对 seed42/45（n=2）。受“只验证双
head”约束，本轮不补训 seed48 scalar baseline，也不能把该差距表述为 3-seed 结论。

### 双 head 正式结果与 Discovery 有效性审计（2026-09-09）

seed42/45/48 的双即时价值 head 验证已完成。三个 seed 的总体 Recovery↓ / Discovery↑ 方向存在，
但连续插值没有优于 single-head：相邻权重 Top-150 Jaccard 从 single-head 的 `0.887` 降至 `0.736`，
且 seed48 在高 Discovery 端出现局部反转。因此连续 preference 不作为当前成立能力；仅冻结
`(0.8,0.2)/(0.5,0.5)/(0.2,0.8)` 三档用于后续审计。

随后执行了 reward 独立的 Discovery 有效性审计。三档候选在读取外部证据前冻结并记录 ranking、
config、checkpoint 与候选文件哈希；审计没有训练、调 reward、增加 seed、重选 checkpoint 或读取
Test 标签。独立证据包括预先冻结的 RCC 文献盲评集、CPTAC ccRCC 独立突变/蛋白组和 DepMap 24Q4
ccRCC CRISPR 依赖。内部 `LowFrequencyEvidenceScoreV2` 只作为直接排序基线和相关性诊断，不作为外部终点。

在共同的 2,419 个低频新候选池内，Top-150 的独立证据命中率 / fold enrichment 为：

| 方法 | 命中率 | Fold | 跨 seed Top-150 Jaccard |
|---|---:|---:|---:|
| mutation | 0.420 | 2.39 | 静态 |
| GCN | 0.389±0.017 | 2.21±0.10 | 0.822 |
| dual-head bandit Recovery-heavy | 0.360±0.013 | 2.05±0.08 | 0.614 |
| MLP | 0.324±0.079 | 1.85±0.45 | 0.604 |
| dual-head bandit Balanced | 0.313±0.035 | 1.78±0.20 | 0.545 |
| GRN degree | 0.287 | 1.63 | 静态 |
| dual-head bandit Discovery-heavy | 0.251±0.089 | 1.43±0.50 | 0.379 |
| PPI degree | 0.187 | 1.06 | 静态 |
| EvidenceScore | 0.133 | 0.76 | 静态 |

三个 Bandit 档位均未在全部 seed 上超过最强简单基线；严格配对 seed42/45 时，GCN 对三个档位均为
2 胜 0 负。剔除 CPTAC 突变复现的事后敏感性分析中，最佳 Bandit 命中率为 `0.080`，仍低于
PPI degree `0.093`、MLP `0.091` 和 GCN `0.089`。Bandit 不是 EvidenceScore 或 degree 的简单复制，
但其低频池排名仍与 mutation 和 PPI degree 中度相关，且 Discovery-heavy 的独立命中与稳定性最弱。

**当前路线判定**：按照预先约定的停止规则，Bandit 不再作为项目主线，仅保留为对照；下一阶段转向
静态融合 / 监督排序。由于本轮 CPTAC、DepMap 和盲评结果已经用于路线判断，它们不得再用于新模型权重、
checkpoint 或候选选择；新路线完成后需要新的未查看外部证据作最终评价。完整冻结协议、逻辑勘误、
逐方法结果和图表位于 `outputs/discovery_validity_audit_20260909/`。
