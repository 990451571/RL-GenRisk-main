#!/usr/bin/env bash
# 双头 vector-Q MORL：公平的 3-seed 验证（不读取 Test 标签）。
# 50 episodes = 5 个已见 preference 各恰好训练 10 次，匹配每个 scalarized
# 对照 run 的 10-episode 暴露量。

set -eu
cd /mnt/e/Projects/RL-GenRisk-main

# Permit an explicit interpreter, so the validation cannot accidentally use a
# system Python outside the project environment.
PY="${PY:-python3}"
EV=/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIRS=()

for seed in 42 45 46; do
  OUTPUT_DIR="outputs/morl_vectorq_popart_50ep_${RUN_ID}_seed${seed}"
  printf '\n[vector-MORL PopArt] starting seed %s/46; output=%s\n' "$seed" "$OUTPUT_DIR"
  "$PY" scripts/train_preference_morl.py \
    --vector-morl \
    --seed "$seed" --max_episodes 50 --feature-mode hybrid6_raw \
    --reward-mode rd_scan --lowfreq-evidence-path "$EV" \
    --epsilon_end 0.05 --per_alpha 0.6 --per_beta_frames auto --gradient_clip 50.0 \
    --output_dir "$OUTPUT_DIR"
  RUN_DIR=$(find "$OUTPUT_DIR/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | sort | tail -n 1)
  if [ -z "$RUN_DIR" ] || [ ! -f "$RUN_DIR/summary.json" ]; then
    printf '[MORL] seed %s completed without an expected summary.json; aborting.\n' "$seed" >&2
    exit 1
  fi
  "$PY" -c 'import json, sys; counts=json.load(open(sys.argv[1], encoding="utf-8"))["trained_preference_counts"]; assert len(counts) == 5 and set(counts.values()) == {10}, counts' "$RUN_DIR/summary.json"
  RUN_DIRS+=("$RUN_DIR")
  printf '[vector-MORL] completed seed %s/46: %s\n' "$seed" "$RUN_DIR"
done

COMPARISON_DIR="outputs/morl_vectorq_popart_50ep_${RUN_ID}_comparison"
printf '\n[vector-MORL] comparing retained MORL checkpoints with scalarized runs...\n'
"$PY" scripts/analyze_preference_morl.py \
  --morl-runs "${RUN_DIRS[@]}" \
  --scalar-summary outputs/rdprobe_rollout_primary_eval_5seed/summary_metrics.csv \
  --output "$COMPARISON_DIR"
RESULT="$COMPARISON_DIR/morl_vs_scalar_frontier_coverage.csv"
if [ ! -s "$RESULT" ]; then
  printf '[MORL] comparison result was not written: %s\n' "$RESULT" >&2
  exit 1
fi
printf '[vector-MORL] comparison completed: %s\n' "$RESULT"

DIAGNOSTIC_DIR="${COMPARISON_DIR}/vector_q_diagnostics"
printf '[vector-MORL] auditing rollout interpolation, head scales, and objective-gradient cosine...\n'
"$PY" scripts/audit_preference_policy_interpolation.py \
  --morl-runs "${RUN_DIRS[@]}" \
  --output "$DIAGNOSTIC_DIR/policy_interpolation"
"$PY" scripts/diagnose_vector_morl.py \
  --morl-runs "${RUN_DIRS[@]}" \
  --output "$DIAGNOSTIC_DIR/learning_diagnostics"
if [ ! -s "$DIAGNOSTIC_DIR/learning_diagnostics/objective_gradient_cosine.csv" ]; then
  printf '[vector-MORL] diagnostics were not written as expected.\n' >&2
  exit 1
fi
printf '[vector-MORL] diagnostics completed: %s\n' "$DIAGNOSTIC_DIR"
