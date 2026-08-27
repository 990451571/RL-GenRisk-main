# RL-GenRisk Agent Handoff Report

生成日期：2026-08-26

用途：这份报告给后续接手的 agent 使用。目标是让 agent 在修改 RL-GenRisk 代码前，先理解当前项目真实结构、关键源码入口、历史 Stage 实验代码、数据边界和已知风险点。

本报告基于当前磁盘状态：

- 主项目：`E:\Projects\RL-GenRisk-main`
- 历史实验归档：`E:\codex_file`
- 当前主项目中已经复制了部分历史实验脚本到 `experiments`

## 1. 工作区状态

当前 Git 工作区不是干净状态。`git status --short` 显示大量已修改、已新增和未跟踪文件，包括 `src/DQN.py`、`src/train.py`、`src/inputall.py`、`src/qfunction.py`、`src/replay_buffer.py`、`src/identify.py`、`src/lowfreq_evidence.py`、`src/mutation_frequency.py`、`scripts/*`、`config/*`、`data/*`、`experiments/*` 等。

后续 agent 修改代码时必须遵守：

- 不要执行 `git reset --hard`。
- 不要用 `git checkout -- <file>` 回滚用户已有修改。
- 不要用历史快照覆盖当前 `src`，除非用户明确要求。
- 修改前先读目标文件当前内容，确认是否已有用户/前序 agent 修改。
- 大型运行产物、checkpoint、`outputs` 不应作为代码修改对象。

2026-08-27 仓库清理记录（均已由用户确认执行）：

- 删除了空占位文件 `src/build_multiomics_features.py`（0 字节，无引用）。
- 删除了历史产物：`src/Ranking_List*.txt`、`src/*_top100.txt`、src 下分析 PDF、`src/agent_ccRCC*.th`、`data/agent_*.th`、`data/*.png` 训练曲线、`data/processed/ppi_overlap_check/`。
- 删除了空文件 `src/test_KIRC.txt`、`src/log/log_KIRC.txt`、空目录 `multi-omics data/`、陈旧示例 `data/processed/multiomics_gene_features_example.csv`、所有 `__pycache__`。
- 删除了 `outputs/` 中的 smoke run 输出和各正式 run 的逐集 ranking CSV；保留了 `outputs/hybrid6_raw_100ep_v1`、`outputs/hybrid6_raw_retrain_v1` 的 checkpoint/summary/best ranking 等关键证据。
- 删除了遗留脚本：`run_in_ubuntu.ps1`、`src/check_kirc_sample_counts.py`、`src/inspect_raw_files.py`、`src/check_multiomics_ppi_overlap.py`（与 `inputall.py` 内同名函数重复）、`src/utils_draw.py`。
- 保留了 `src/evaluate_frozen_test_cpu_deterministic.py`（CPU 确定性复现 harness，依赖 `E:\codex_file` 归档，非重复文件）。
- 上述被删除的文件均为 git 跟踪内容，可通过 git 历史恢复；未提交清理提交。

2026-08-27 死代码清理（已由用户确认）：

- `src/DQN.py`：删除 12 个 TF 时代死方法（`getBatch`、`store_transition`、`getState`、`getAction`、`get_train_Q`、`choose_action`、`getQt`、`laplacian`、`get_feature`、`Normalized_minmax`、`Normalized`、`plot_*`），删除 `getAcc` 及 step() 内无人读取的 train/test 覆盖率块，删除 agent 级 `save/load/save_checkpoint/load_checkpoint`，删除死属性（`n_step`、`a_ori`、`gene_ori`、`train_sta`、`train_before`、`cost_his*`、`train_cover`、`test_cover`、`epsilon_increment`）与死参数（`e_greedy`、`replace_target_iter`、`output_graph`、`e_greedy_increment`），移除 matplotlib/sklearn 导入。
- `src/train.py`：删除无调用的 `load_hybrid6_raw_features`（92 行，与 `load_node_features_by_mode` 重复）、`set_random_seed`、未使用常量 `FEATURE_COLUMNS`/`FEATURE_DIM`。
- `src/inputall.py`：删除无调用的 `get_node_feature_report_by_mode`、`random_getGene`、`random_patient`。
- `src/identify.py`：构造调用去掉 `e_greedy`/`replace_target_iter` 参数。
- 注意：`experiments/scripts/stage4*.py` 中残留 `agent.epsilon_increment = 0.0` 一行（无害的实例属性赋值，因 learn() 已不再更新 epsilon；实验脚本为冻结历史，未改动）。
- `experiments/scripts/stage4*.py` 以 `Stage4DeepQNetwork(DeepQNetwork)` 子类化，只覆写 `__init__/reward_config/step`，依赖基类 `get_reward/step/_compose_reward_components/learn/remember`——均已保留。
- 三个 evaluate 脚本（frozen_test / cpu_deterministic / external_holdout）与 `aggregate_final_evaluation.py` 属于已冻结最终评估协议（SHA-256 记录于 `E:\codex_file\二阶段\02_final_evaluation\00_freeze_audit\evaluation_code_hashes.csv`），保持不动，禁止合并或修改。

## 2. 当前顶层目录

`E:\Projects\RL-GenRisk-main` 当前主要内容：

- `src`：当前主项目源码。正式训练、模型、数据加载、reward、DDQN、PER、评价和 checkpoint 逻辑都在这里。
- `scripts`：项目辅助脚本，例如 CNV 构建、健康检查、突变频率审计。
- `experiments`：从 `E:\codex_file` 复制来的 Stage1-4B 历史实验脚本和配置，用于后续开发参考或纳入 Git 管理。
- `data`：项目数据、旧 checkpoint、PPI、突变、多组学处理结果、CNV 处理结果等。
- `config`：本地路径配置。`config/local_paths.yaml` 是机器本地配置，已被 `.gitignore` 忽略。
- `outputs`：训练和 smoke run 输出。不要覆盖已有结果。
- `backup`：历史备份。不要当作当前源码入口。
- `multi-omics data`：当前检查未列出文件，可能为空或无普通文件。

`README.md` 是项目简要入口；更详细的 agent 交接说明在本文件。

## 3. 主项目核心源码

### `src/train.py`

当前主训练入口。

关键函数：

- `parse_args()`：命令行参数定义，包括 `--feature-mode`、`--reward-mode`、label 路径、checkpoint/resume、训练超参。
- `validate_training_args(args)`：检查 reward mode、lowfreq evidence 参数、标签路径等。
- `build_environment(args, run_dir, normalization_metadata=None)`：加载 mutation、PPI、weights、gene universe、node features，并返回训练环境字典。
- `build_agent(args, env, device)`：构建 `DeepQNetwork`。
- `run_episode(agent, env, args, episode, run_dir=None)`：episode 主循环，执行 action、reward、replay buffer 写入和学习。
- `evaluate_validation(agent, env, args, run_dir, episode)`：生成 validation ranking，计算 metrics。
- `write_ranking(path, q_values, gene_name, feature_mode=FEATURE_MODE)`：把模型 Q 值转成 gene ranking。
- `checkpoint_payload(...)`、`save_checkpoint(...)`、`load_checkpoint(...)`：checkpoint 保存和读取。
- `main()`：正式训练主入口。

修改建议：

- 改训练流程、评估频率、checkpoint 选择规则：优先看 `main()`、`evaluate_validation()`、`save_checkpoint()`。
- 改输入特征：优先看 `FEATURE_MODE_COLUMNS` 的来源 `inputall.FEATURE_MODE_COLUMNS`，以及 `load_node_features_by_mode()` / `build_environment()`。
- 改 reward mode 参数：先改 `REWARD_MODES` 和 `validate_training_args()`，再改 `src/DQN.py` 的 reward 组合逻辑。

### `src/DQN.py`

当前 DDQN agent 和 reward 主体。

关键类：

- `DeepQNetwork`

关键逻辑：

- `__init__()`：构建 online Q 网络 `self.Q` 和 target Q 网络 `self.Q_target`，都使用 `Q_Fun`。
- `_compose_reward_components(...)`：组合 legacy、多组学、lowfreq reward 分量。
- `get_reward(...)`：legacy reward 的部分基础计算。
- `step(...)`：执行选择基因后的环境状态更新、reward 计算、done 判断。
- `learn()`：PER 采样、DDQN target、loss、反向传播、priority update、target soft update。
- checkpoint 保存/读取统一由 `src/train.py` 模块级 `save_checkpoint()` / `load_checkpoint()` 承担（agent 级 checkpoint 方法已于 2026-08-27 删除）。

DDQN 实现位置：

- Online network 选择 next action：`learn()` 中 `next_q_values_online.argmax(...)`。
- Target network 评价该 action：`learn()` 中 `self.Q_target(...)` 后 `gather(1, best_next_actions)`。
- TD error：`td_errors = y_target - y_pred`。
- PER priority update：`self.memory.update_priorities(sample_indices, td_errors_np)`。
- Soft update：遍历 `self.Q_target.parameters()` 和 `self.Q.parameters()`，用 `tau` 更新 target。

修改建议：

- 改 DDQN target 或 target update：只改 `learn()`，并同步检查 `replay_buffer.py` 的采样返回顺序。
- 改 reward：优先改 `_compose_reward_components()` 和 `step()`，不要在 `train.py` 里临时拼 reward。
- 改 action mask：先读 `run_episode()` 如何传 `current_action_mask` / `next_action_mask`，再改 `learn()` 中 mask target 的部分。

### `src/qfunction.py`

当前主项目 Q 网络结构。

关键内容：

- `change(matrix)`：把 dense adjacency matrix 转成 PyG `edge_index`。
- `class Q_Fun(nn.Module)`：GCN + Q 输出网络。

当前主项目 `qfunction.py` 本身只接收一个 adjacency matrix。它没有显式区分 PPI 和 GRN。Stage2/4/4B 的 PPI+GRN 是在外层实验脚本中把 combined message graph matrix 传入 `Q_Fun`，不是在主 `qfunction.py` 中写死多关系逻辑。

修改建议：

- 如果要做真正多关系 GNN，不建议直接把当前 `Q_Fun` 改复杂；先参考 `experiments/scripts/stage3_relation_aware_experiment.py` 里的 `DualBranchGlobalGateQ`。
- 如果只做 SimpleUnion，优先在外层构建 combined adjacency，不必改 `Q_Fun`。

### `src/replay_buffer.py`

PER buffer。

关键类：

- `PrioritizedReplayBuffer`

关键函数：

- `store_transition(...)`：写入经验和初始 priority。
- `sample_buffer(batch_size)`：按 priority 采样，返回 importance sampling weights。
- `update_priorities(indices, td_errors)`：用 TD error 更新 priority。

修改建议：

- 改 PER alpha/beta 只需要改训练参数或默认值。
- 改采样字段顺序时，必须同步修改 `DQN.learn()` 解包顺序。
- 不要把 validation/test label 放进 replay buffer。

### `src/inputall.py`

数据加载和 node feature 构建。

关键函数：

- `FEATURE_MODE_COLUMNS`：定义各 feature mode 的列。
- `load_multiomics_features(...)` / `load_multiomics_features_for_columns(...)`：读取多组学特征。
- `get_node_features_by_mode(...)`：根据 feature mode 生成节点特征。
- `getInput(...)`：读取 mutation/patient-gene 输入。
- `getWeight(...)`：读取 weights。
- `getNetwork(...)` / `getNetworkall(...)`：读取 PPI/network 文件并生成 adjacency。

修改建议：

- 新增特征模式：先加 `FEATURE_MODE_COLUMNS`，再检查 `get_node_features_by_mode()`、`train.py` 的 `FEATURE_DIM_BY_MODE`。
- 改 CNV 处理：优先检查 `scripts/build_kirc_cnv_feature.py` 和 `data/processed/cnv_kirc`。
- 改 gene universe：风险很高，会影响 checkpoint、ranking、label 对齐、edge index、feature matrix。必须同步审计所有 9039 假设。

### `src/mutation_frequency.py`

突变频率分层工具。

关键函数：

- `classify_mutation_frequency(patient_count)`
- `mutation_frequency(patient_count, total_samples=TOTAL_KIRC_TUMOR_SAMPLES)`
- `mutation_frequency_pct(patient_count, total_samples=TOTAL_KIRC_TUMOR_SAMPLES)`

当前 Stage4-B 使用这类逻辑区分 `very_low`、`low_frequency`、`high_frequency`。修改突变分层时，要同步检查 Stage4/Stage4B reward 设计脚本。

### `src/lowfreq_evidence.py`

低频 evidence table 读取工具。

关键函数：

- `load_evidence_table(path)`
- `load_evidence_by_gene(path, gene_order=None)`

修改低频 evidence schema 时必须同步：

- `src/lowfreq_evidence.py`
- `src/DQN.py`
- `src/train.py`
- `experiments/scripts/stage4b_missingness_aware_lowfreq.py`

## 4. 当前数据和配置

主数据目录 `data` 当前包含：

- `data/HPRD.txt`：主 PPI 文件。
- `data/GeneID.csv`：基因 ID/基因名来源之一。
- `data/KIRC.txt`：项目默认 mutation/patient-gene 文件。
- `data/raw/KIRC_mc3.txt`：Stage4-B mutation absence/low-frequency 逻辑使用的 raw MC3 文件。
- `data/weights.txt`：训练 reward/weight 相关输入。
- `data/processed/KIRC_multiomics_3omics.csv`：3 组学特征。
- `data/processed/KIRC_multiomics_4omics.csv`：4 组学特征。
- `data/processed/cnv_kirc/KIRC_cnv_gene_feature.csv`：CNV 特征。
- 多个 `.th` checkpoint 和历史 ranking 文本。

`config/local_paths.yaml` 当前使用项目内 Protocol B 标签路径：

- `project_root: /mnt/e/Projects/RL-GenRisk-main`
- `data_root: data`
- `codex_output_root: /mnt/e/codex_file`
- `train_label_path: experiments/protocol_B/train_driver_genes.csv`
- `val_label_path: experiments/protocol_B/validation_driver_genes.csv`

Protocol B 的小型标签 CSV 已复制到 `experiments/protocol_B`，默认 Train/Validation 标签路径不再依赖 `E:\codex_file`。`codex_output_root` 仍保留为历史实验输出归档位置。

## 5. 已复制的历史实验文件

当前采用扁平结构：

- `experiments/scripts/stage1_grn_audit.py`
- `experiments/scripts/stage2_multigraph_experiment.py`
- `experiments/scripts/stage3_relation_aware_experiment.py`
- `experiments/scripts/stage4_fixed_preference.py`
- `experiments/scripts/stage4b_missingness_aware_lowfreq.py`
- `experiments/configs/stage2_formal_config.yaml`
- `experiments/configs/stage3_formal_config.yaml`
- `experiments/configs/stage4_formal_config.yaml`
- `experiments/configs/stage4b_formal_config.yaml`
- `experiments/configs/stage4b_fixed_preferences.yaml`
- `experiments/protocol_B/generate_driver_label_protocol.py`
- `experiments/protocol_B/evaluate_gene_ranking.py`
- `experiments/protocol_B/evaluation_unit_test.py`
- `experiments/protocol_B/train_driver_genes.csv`
- `experiments/protocol_B/validation_driver_genes.csv`
- `experiments/protocol_B/test_driver_genes.csv`
- `experiments/protocol_B/sensitivity_shared_external.csv`
- `experiments/protocol_B/split_summary.json`

已确认：

- `experiments/protocol_B/split_summary.json` 当前是 Protocol B，内容显示 `"protocol": "protocol_B"`。
- `experiments/protocol_B` 下的 Train/Validation/Test/Sensitivity CSV 已复制进主项目。
- `experiments/configs/stage4_formal_config.yaml` 当前是 Stage4 formal config，内容包含 `STAGE4_BACKBONE: "Stage2_SimpleUnion"`。
- `experiments/scripts/stage1_grn_audit.py` 等主线脚本和源 `E:\codex_file` 文件哈希一致。

## 6. Stage 实验含义

### Stage1 GRN

源码：

- `experiments/scripts/stage1_grn_audit.py`
- 原始位置：`E:\codex_file\新方向阶段1\scripts\stage1_grn_audit.py`

关键函数：

- `download_grn()`
- `parse_and_clean_grn(raw_path, universe)`
- `stage2_ready(...)`

含义：

- 从 OmniPath interactions API 获取 DoRothEA GRN。
- 阈值使用 A+B 作为 primary。
- 把 GRN 映射到 9039 gene universe。
- 输出 Stage2 可用的 frozen PPI/GRN/gene universe 文件到历史目录。

### Stage2 PPI/GRN/SimpleUnion

源码：

- `experiments/scripts/stage2_multigraph_experiment.py`

关键函数：

- `generate_message_graphs(frozen)`
- `matrix_from_message_edges(path, node_count=9039)`
- `build_run(condition, seed, run_dir, message_edge_path, smoke=False)`
- `run_formal(message_paths)`

核心逻辑：

- PPI-only：PPI bidirectional message graph。
- GRN-only：DoRothEA A+B directed TF-to-target message graph。
- PPI+GRN SimpleUnion：PPI 双向边和 GRN 有向边做 directed edge union。
- 特征和 action topology 仍来自原始 PPI 环境，Stage2 wrapper 只替换传给 GCN 的 message graph matrix。

### Stage3 RelationAware

源码：

- `experiments/scripts/stage3_relation_aware_experiment.py`

关键类：

- `DualBranchGlobalGateQ`

关键结构：

- PPI branch：`ppi_lin1`、`ppi_conv1`、`ppi_conv2`、`ppi_lin2`
- GRN branch：`grn_lin1`、`grn_conv1`、`grn_conv2`、`grn_lin2`
- Global gate：`relation_logits` 经过 softmax，融合 PPI/GRN branch embedding。

用途：

- 验证 relation-aware encoder 是否优于 Stage2 SimpleUnion。
- Stage3 最终决策中，Stage4 backbone 使用 `Stage2_SimpleUnion`。

### Stage4-A MORL

源码：

- `experiments/scripts/stage4_fixed_preference.py`

关键类：

- `Stage4RewardModel`
- `Stage4DeepQNetwork`

关键函数：

- `load_stage4_reward_model()`
- `build_stage4_agent(...)`
- `run_episode_stage4(...)`
- `build_run(...)`
- `run_formal(...)`
- `analyze(...)`

含义：

- 固定偏好向量：Recovery、Discovery、Robustness。
- Backbone 明确写为 `Stage2_SimpleUnion`。
- 使用 PPI+GRN SimpleUnion message graph。
- Validation labels 不进入 reward，只用于 checkpoint/metrics。

### Stage4-B Missingness-Aware Low-Frequency Reward

源码：

- `experiments/scripts/stage4b_missingness_aware_lowfreq.py`

关键类：

- `Stage4RewardModel`
- `Stage4DeepQNetwork`

关键函数：

- `low_frequency_target_score(count, tau=LOW_FREQUENCY_TARGET_TAU)`
- `load_raw_mutation_gene_sets()`
- `mutation_evidence_status(gene, count, raw_any_tumor_genes)`
- `write_reward_design_files(model)`
- `run_reward_unit_tests(model, score_rows)`
- `run_formal(model)`
- `analyze(model)`

含义：

- 继承 Stage2_SimpleUnion backbone。
- 修改 Discovery reward 中突变 target 逻辑。
- 不把 N=0 当成确认低频突变证据。
- `N0_MUTATION_TARGET_REWARD = 0.0`
- `N1_MUTATION_TARGET_REWARD = 0.5`
- `LOW_FREQUENCY_TARGET_RANGE = 2 <= N <= 18`
- 高于 18 使用指数衰减。

## 7. 当前主运行调用链

一般训练入口：

1. `src/train.py::main()`
2. `parse_args()`
3. `validate_training_args(args)`
4. `choose_device(args.device)`
5. `make_run_dir(...)`
6. `build_environment(args, run_dir, normalization_metadata=None)`
7. `inputall.getInput(...)`
8. `inputall.getWeight(...)`
9. `inputall.getNetwork(...)`
10. `load_node_features_by_mode(...)`
11. `build_agent(args, env, device)`
12. `DQN.DeepQNetwork(...)`
13. `qfunction.Q_Fun(...)`
14. episode loop：`run_episode(...)`
15. action selection：epsilon-greedy + action mask
16. `agent.step(...)`
17. reward components
18. `agent.remember(...)`
19. `optimize_model(...)`
20. `agent.learn()`
21. PER sample
22. DDQN target
23. backward + optimizer step
24. PER priority update
25. target soft update
26. validation：`evaluate_validation(...)`
27. ranking：`write_ranking(...)`
28. checkpoint：`save_checkpoint(...)`

## 8. GRN 接入事实

当前主项目 `src/qfunction.py` 没有直接写 PPI/GRN 双关系。它只把传入 adjacency matrix 转成 `edge_index`。

GRN 实验真实接入位置在历史 Stage 脚本：

- Stage1 构建 GRN：`experiments/scripts/stage1_grn_audit.py`
- Stage2 SimpleUnion：`experiments/scripts/stage2_multigraph_experiment.py`
- Stage3 DualBranch：`experiments/scripts/stage3_relation_aware_experiment.py`
- Stage4/4B 使用 Stage2 SimpleUnion message graph：`experiments/scripts/stage4_fixed_preference.py`、`experiments/scripts/stage4b_missingness_aware_lowfreq.py`

因此，后续若要继续 GRN 开发，有两条路线：

- SimpleUnion 路线：继续在外层构建 combined message graph，传给当前 `Q_Fun`。
- RelationAware 路线：把 `DualBranchGlobalGateQ` 从 Stage3 实验脚本整理进主项目模块，但要同时处理 checkpoint schema、training import、parameter count、masking/evaluation。

## 9. Train / Validation / Test 边界

当前实验协议强调：

- Train labels 可用于 reward 或训练信号。
- Validation labels 用于 checkpoint selection 和 metrics。
- Test labels 不应在训练、reward、checkpoint selection 中读取。
- Historical Test 和 external holdout 不应提前读取。

Protocol B 相关：

- 当前主项目已包含 `experiments/protocol_B/split_summary.json`。
- 当前主项目已包含 `train_driver_genes.csv`、`validation_driver_genes.csv`、`test_driver_genes.csv`、`sensitivity_shared_external.csv`。
- `config/local_paths.yaml` 和 `config/local_paths.example.yaml` 当前指向项目内 Train/Validation CSV。

## 10. 常见修改任务入口

改低频突变分层：

- `src/mutation_frequency.py`
- `scripts/audit_mutation_frequency_redefinition.py`
- `experiments/scripts/stage4b_missingness_aware_lowfreq.py`

改 low-frequency discovery reward：

- `src/DQN.py`
- `src/lowfreq_evidence.py`
- `src/train.py`
- `experiments/scripts/stage4b_missingness_aware_lowfreq.py`

改 GCN/GNN 结构：

- Simple path：`src/qfunction.py`
- Relation-aware path：参考 `experiments/scripts/stage3_relation_aware_experiment.py::DualBranchGlobalGateQ`

改 PPI/GRN message graph：

- `experiments/scripts/stage2_multigraph_experiment.py::generate_message_graphs`
- `experiments/scripts/stage2_multigraph_experiment.py::matrix_from_message_edges`
- `experiments/scripts/stage4_fixed_preference.py::matrix_from_message_edges`
- `experiments/scripts/stage4b_missingness_aware_lowfreq.py::matrix_from_message_edges`

改 node features：

- `src/inputall.py::FEATURE_MODE_COLUMNS`
- `src/inputall.py::get_node_features_by_mode`
- `src/train.py::load_node_features_by_mode`
- `src/train.py::build_environment`

改 DDQN/PER：

- `src/DQN.py::learn`
- `src/replay_buffer.py::PrioritizedReplayBuffer`

改 ranking/evaluation：

- `src/train.py::write_ranking`
- `src/train.py::metrics_at_k`
- `src/train.py::evaluate_validation`
- `experiments/protocol_B/evaluate_gene_ranking.py`

改 checkpoint：

- `src/train.py::checkpoint_payload`
- `src/train.py::save_checkpoint`
- `src/train.py::load_checkpoint`

## 11. 当前已知风险和待处理事项

1. 空占位文件 `src/build_multiomics_features.py` 已于 2026-08-27 删除。

   它曾是 0 字节占位文件且无任何代码引用。重建多组学特征的正规入口：
   - 3 组学（Gene/Mutation/Expression/Methylation）：`src/process_kirc_3omics.py`
   - CNV 与 4 组学（追加 CNV 列）：`scripts/build_kirc_cnv_feature.py`

2. 主项目和历史 Stage 脚本属于不同层级。

   主项目 `src` 是当前运行代码；`experiments/scripts` 是历史实验 wrapper 和设计证据。不要假设二者完全同步。

3. 当前 `experiments` 尚未被 Git 跟踪。

   `git status --short` 显示 `?? experiments/`。如果用户想用 Git 保存 Stage，需要先 `git add experiments` 并提交。

4. 大量 checkpoint 和输出文件存在于 `data`、`src`、`outputs`。

   修改时不要覆盖 `.th`、`.pt`、ranking、validation metrics、formal outputs。

5. 9039 gene universe 是多处隐含假设。

   Stage1/Stage2/Stage4/4B 脚本和主训练代码都围绕 9039 节点构建。改 gene universe 会影响 edge_index、feature matrix、ranking、label overlap 和 checkpoint 兼容性。

6. 排名方法当前是「一次性打分」，可考虑改进为「贪心 rollout」。

   当前 `evaluate_validation` / `write_ranking` 用空选择（mask 全 1）做单次前向，按每个基因在该状态下的 Q 值排序。这在训练第 0 步（同样空选择）下是自洽的，不是 bug。但它没有利用模型「逐个选」的能力去考虑基因之间的冗余/互补。

   未来可改进方向（研究性，非 bug 修复）：改成贪心 rollout —— 选最高 Q 的基因 → 更新动作掩码 → 基于新状态重新打分 → 再选下一个，直到达到 topk，得到一个考虑冗余的排序。涉及 `src/train.py::evaluate_validation`、`write_ranking` 与 `src/qfunction.py::Q_Fun.forward` 的 graph_pool 上下文。

## 12. 给后续 agent 的建议流程

接到新需求后建议按此顺序：

1. 先读本报告。
2. 跑只读状态检查：`git status --short`。
3. 明确需求属于哪类：reward、feature、GRN/GNN、DDQN/PER、evaluation、checkpoint、data protocol。
4. 读取对应源码，不要直接凭历史报告改。
5. 如果涉及 Stage4/4B，先确认 `experiments/configs/stage4_formal_config.yaml` 仍是 formal config，而不是偏好文件。
6. 如果涉及标签路径，优先使用 `experiments/protocol_B` 内的 Protocol B CSV。
7. 修改尽量集中，避免顺手重构。
8. 修改后优先跑轻量审计/单元检查，再跑 smoke；正式训练必须等用户明确授权。
9. 不要读取 Test label，除非任务明确是 final test evaluation。
10. 提交前总结改了哪些文件、为何改、如何验证。

## 13. 推荐 Git 保存策略

如果用户希望用 Git 保存不同 Stage：

- 已完成且不再变的 Stage 用 tag。
- 仍会继续开发的 Stage 用 branch。
- 当前 `main` 保留最新主线。
- 不要长期靠 `src_snapshot` 保存版本；快照只作为历史证据。

建议 tag 名：

- `stage1-grn-freeze`
- `stage2-simpleunion-freeze`
- `stage3-relationaware-freeze`
- `stage4-morl-freeze`
- `stage4b-lowfreq-reward-freeze`

前提：先把 `experiments` 中已复制的 Stage 脚本和配置纳入 Git。

## 14. 结论

RL-GenRisk 当前主项目的可修改核心仍在 `src`。`experiments` 目录保存的是从 `E:\codex_file` 迁移来的 Stage1-4B 历史实验逻辑，主要用于复现、审计和指导后续开发。

后续 agent 最容易犯的错误是：

- 把 Stage 实验 wrapper 当成当前主训练入口。
- 用 Stage4 快照覆盖当前 `src`。
- 把 `codex_output_root` 的历史归档路径误当成默认标签路径。
- 把 `stage4_formal_config.yaml` 和 `stage4_fixed_preferences.yaml` 混用。
- 在 reward 或 checkpoint 选择中误读 Validation/Test label。

修改代码前必须先定位需求属于哪条链路，并只改对应模块。
