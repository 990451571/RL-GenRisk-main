# RL-GenRisk

RL-GenRisk is a research codebase for ranking cancer-related genes with multi-omics data, biological networks, graph neural networks, and DDQN/PER reinforcement learning.

This repository is currently an experimental research workspace, not a packaged clinical or production tool. Rankings produced by the code are research outputs and must not be interpreted as clinical risk conclusions.

## Main Entry Points

- `src/train.py`: main training, validation, ranking, and checkpoint flow.
- `src/DQN.py`: DDQN agent, reward components, PER learning update, and target-network update.
- `src/qfunction.py`: GCN-based Q function used by the current main training path.
- `src/inputall.py`: data loading and node-feature construction.
- `src/replay_buffer.py`: prioritized replay buffer.
- `src/mutation_frequency.py`: mutation-frequency stratification utilities.

## Experiment Code

Historical staged experiment code is kept in a flat `experiments` layout:

- `experiments/scripts/stage1_grn_audit.py`
- `experiments/scripts/stage2_multigraph_experiment.py`
- `experiments/scripts/stage3_relation_aware_experiment.py`
- `experiments/scripts/stage4_fixed_preference.py`
- `experiments/scripts/stage4b_missingness_aware_lowfreq.py`
- `experiments/configs`
- `experiments/protocol_B`

Read `AGENT_HANDOFF_REPORT.md` before making code changes.

## Data Protocol

Protocol B label files are stored under `experiments/protocol_B` so the default Train/Validation label paths do not depend on `E:\codex_file`.

The current method uses Train labels during training/reward where configured, Validation labels for checkpoint selection and metrics, and Test labels only for explicitly requested final test evaluation.

## Environment

Use `environment.yaml` as the primary environment specification. `requirements.txt` is not sufficient by itself for model training because the project depends on PyTorch and PyTorch Geometric.

## Current Method Note

The current main training path selects from the global 9039-gene action pool with an action mask. This differs from a strict PPI-neighborhood expansion MDP. Treat any change back to neighbor-restricted actions as a separate experimental stage.
