# Discovery 有效性审计

生成时间（UTC）：2026-09-09T07:40:02.584069+00:00

## 结论

按预先冻结并经逻辑勘误的严格规则，没有 Bandit 策略在 3 个 seed 上均超过最强 mutation/EvidenceScore/degree 简单基线；当前证据不支持继续把 Bandit 作为主线，应转向静态融合/监督排序。

这是方法路线结论，不是新癌基因或临床有效性结论。

## 事实边界

- 全基因宇宙：9039；低频新候选池：2419。
- 低频候选池内冻结文献盲评阳性：8/16（其余盲评基因为 N<2，不属于本轮固定低频定义）。
- 低频候选池内 CPTAC 独立复现阳性：290；CPTAC 蛋白支持：120。
- 低频候选池内 DepMap ccRCC 依赖：313；其中选择性依赖：22。
- Test 标签未读取；没有训练、checkpoint 重选、reward 调整、模型修改或新增 seed。
- GRN degree 只用于事后混杂审计，不是最新 Bandit 的输入特征。

## 主要比较（共同低频候选池 Top-150）

| 方法 | n | 独立证据命中率 mean±SD | Fold mean±SD |
|---|---:|---:|---:|
| mutation | 1 | 0.420±0.000 | 2.39±0.00 |
| GCN | 3 | 0.389±0.017 | 2.21±0.10 |
| bandit_recovery_heavy | 3 | 0.360±0.013 | 2.05±0.08 |
| MLP | 3 | 0.324±0.079 | 1.85±0.45 |
| bandit_compromise | 3 | 0.313±0.035 | 1.78±0.20 |
| GRN_degree | 1 | 0.287±0.000 | 1.63±0.00 |
| bandit_discovery_heavy | 3 | 0.251±0.089 | 1.43±0.50 |
| PPI_degree | 1 | 0.187±0.000 | 1.06±0.00 |
| EvidenceScore | 1 | 0.133±0.000 | 0.76±0.00 |

严格 Bandit 判定：
- bandit_compromise: 0.313±0.035; 三个 seed 均高于最强简单基线=false。
- bandit_discovery_heavy: 0.251±0.089; 三个 seed 均高于最强简单基线=false。
- bandit_recovery_heavy: 0.360±0.013; 三个 seed 均高于最强简单基线=false。

## 事后敏感性分析：剔除 CPTAC 突变复现

该分析只检查主结果是否被独立队列突变频率主导，不用于重新选择模型。

| 方法 | 非突变外部证据命中率 | Fold |
|---|---:|---:|
| PPI_degree | 0.093 | 1.51 |
| MLP | 0.091 | 1.47 |
| GCN | 0.089 | 1.43 |
| EvidenceScore | 0.080 | 1.29 |
| bandit_recovery_heavy | 0.080 | 1.29 |
| bandit_discovery_heavy | 0.078 | 1.25 |
| mutation | 0.073 | 1.18 |
| bandit_compromise | 0.067 | 1.08 |
| GRN_degree | 0.053 | 0.86 |

## 与监督模型的严格配对比较（仅 seed42/45）

- bandit_compromise vs GCN: 命中率差 mean=-0.077，两 seed 胜/负=0/2。
- bandit_compromise vs MLP: 命中率差 mean=-0.050，两 seed 胜/负=0/2。
- bandit_discovery_heavy vs GCN: 命中率差 mean=-0.120，两 seed 胜/负=0/2。
- bandit_discovery_heavy vs MLP: 命中率差 mean=-0.093，两 seed 胜/负=0/2。
- bandit_recovery_heavy vs GCN: 命中率差 mean=-0.030，两 seed 胜/负=0/2。
- bandit_recovery_heavy vs MLP: 命中率差 mean=-0.003，两 seed 胜/负=1/1。

## 是否只是简单分数重排

- 触发高相似启发式的 Bandit–baseline 配对：0/36。
- 阈值为 |Spearman|≥0.80 或 Top-150 Jaccard≥0.67；它只是诊断阈值，不是统计学等价证明。
- 没有配对达到高相似阈值；但这不自动证明 Bandit 学到了新的生物机制。

## 跨 seed 稳定性

- GCN: mean pairwise Jaccard=0.822（3 seeds）。
- MLP: mean pairwise Jaccard=0.604（3 seeds）。
- bandit_compromise: mean pairwise Jaccard=0.545（3 seeds）。
- bandit_discovery_heavy: mean pairwise Jaccard=0.379（3 seeds）。
- bandit_recovery_heavy: mean pairwise Jaccard=0.614（3 seeds）。

## 限制与风险

- 16 基因盲评集很小且为人工阳性集合；它可检验召回，不能估计完整假阳性率。
- DepMap 是细胞系 CRISPR 必需性，偏向可增殖细胞和必需基因；不等同于患者肿瘤驱动作用。
- CPTAC 蛋白差异说明肿瘤相关表达改变，不等同于致癌因果性；复现突变也受约 100 例队列功效限制。
- MLP/GCN 为 seed42/45/46，Bandit 为 seed42/45/48；三 seed 均值只能描述。严格配对只能使用 seed42/45。
- EvidenceScore 同时参与 Discovery reward，因此只能作为直接排序基线和混杂诊断，绝不能作为 Bandit 的独立有效性终点。

## 输出索引

- `audit_protocol.json`：结果前冻结的口径。
- `audit_protocol_amendment_01.json`：将 mutation 纳入最强简单基线的逻辑勘误；没有改动任何结果阈值。
- `frozen_candidates/freeze_manifest.json`：三档候选、源 ranking/config/checkpoint 哈希。
- `score_feature_spearman.csv`：分数与突变、PPI/GRN degree、多组学相关。
- `top150_reordering_audit.csv`：Bandit 与简单排序的重合/重排诊断。
- `independent_evidence_metrics_per_run.csv` / `independent_evidence_summary.csv`：命中率与富集。
- `paired_seed42_45_comparison.csv`：Bandit 与 MLP/GCN 的同 seed 严格配对。
- `topk_stability.csv`：跨 seed Jaccard。
- `independent_evidence_gene_table.csv`：低频候选的外部证据明细。
