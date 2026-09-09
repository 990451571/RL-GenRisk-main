#!/usr/bin/env bash
# Small 3-seed validation of the dual-head immediate contextual bandit.
set -eu
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${PY:-python3}"
export PYTHONUNBUFFERED=1
EV=/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIRS=()

validate_run() {
  "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1],encoding="utf-8")); c=json.load(open(sys.argv[2],encoding="utf-8")); assert s["status"]=="COMPLETED" and s["bootstrap"] is False and s["action_scalarization"]=="raw_linear" and s["test_labels_read"] is False; assert c["model"]=="dual_head_preference_contextual_bandit" and c["objective_dim"]==2 and c["q_preference_dim"]==0 and c["learning_mode"]=="contextual_bandit" and c["bootstrap"] is False and c["action_scalarization"]=="raw_linear" and c["schedule_policy"]=="balanced_latin_blocks_v1" and set(c["trained_preference_counts"].values())=={10} and all(v==[2,2,2,2,2] for v in c["trained_preference_phase_counts"].values()) and c["test_labels_read"] is False' "$1/summary.json" "$1/dual_head_bandit_config.json"
}

for seed in 42 45 48; do
  output="outputs/dual_head_bandit_${RUN_ID}_seed${seed}"
  completed=""
  if [ -d "$output/hybrid6_raw" ]; then
    completed="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
  fi
  if [ -n "$completed" ] && validate_run "$completed"; then
    printf '\n[dual-head bandit] reuse seed %s: %s\n' "$seed" "$completed"
  else
    if [ -n "$completed" ]; then
      printf '\n[dual-head bandit] incompatible completed run will not be reused: %s\n' "$completed"
    fi
    printf '\n[dual-head bandit] start seed %s: %s\n' "$seed" "$output"
    "$PY" scripts/train_dual_head_bandit.py \
      --seed "$seed" --max_episodes 50 --feature-mode hybrid6_raw \
      --history-ablation full --reward-mode rd_scan \
      --lowfreq-evidence-path "$EV" --epsilon_end 0.05 --per_alpha 0.6 \
      --per_beta_frames auto --gradient_clip 50.0 --output_dir "$output"
    completed="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
    test -n "$completed"
    validate_run "$completed"
  fi
  RUN_DIRS+=("$completed")
done

comparison="outputs/dual_head_bandit_${RUN_ID}_comparison"
"$PY" scripts/analyze_dual_head_bandit.py \
  --dual-runs "${RUN_DIRS[@]}" \
  --expected-seeds 42,45,48 \
  --single-shared-interpolation outputs/preference_bandit_20260908_190650_interpolation \
  --scalar-bandit-eval outputs/rlnecessity_bandit_20260907_152513_eval \
  --output "$comparison"
test -s "$comparison/interpolation_smoothness_comparison.csv"
printf '\n[dual-head contextual bandit] completed: %s\n' "$comparison"
