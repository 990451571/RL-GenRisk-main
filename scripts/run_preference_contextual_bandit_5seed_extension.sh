#!/usr/bin/env bash
# Extend the phase-balanced shared preference bandit from the validated
# 42/45/46 run to five seeds.  This is a shared-model reproducibility test;
# it deliberately does not create unmatched new DDQN/MLP/GCN baselines.
set -eu
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${PY:-python3}"
export PYTHONUNBUFFERED=1
EV=/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv
# The 42/45/46 reference runs are the phase-balanced, 3-seed validation.
REFERENCE_RUN_ID="${REFERENCE_RUN_ID:-20260908_171404}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REFERENCE_RUNS=()
NEW_RUNS=()

validate_run() {
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); c=json.load(open(sys.argv[2],encoding="utf-8")); assert d["status"] == "COMPLETED"; assert c["learning_mode"] == "contextual_bandit" and c["td_target"] == "immediate_reward_only" and c["schedule_policy"] == "balanced_latin_blocks_v1"; assert set(c["trained_preference_counts"].values()) == {10}; assert all(v == [2,2,2,2,2] for v in c["trained_preference_phase_counts"].values())' "$1/summary.json" "$1/preference_bandit_config.json"
}

for seed in 42 45 46; do
  root="outputs/preference_bandit_${REFERENCE_RUN_ID}_seed${seed}/hybrid6_raw"
  run="$(find "$root" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
  test -n "$run"
  validate_run "$run"
  REFERENCE_RUNS+=("$run")
done

for seed in 47 48; do
  output="outputs/preference_bandit_${RUN_ID}_seed${seed}"
  completed=""
  if [ -d "$output/hybrid6_raw" ]; then
    completed="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
  fi
  if [ -n "$completed" ] && validate_run "$completed"; then
    printf '\n[preference-bandit 5-seed extension] reuse seed %s: %s\n' "$seed" "$completed"
  else
    if [ -n "$completed" ]; then
      printf '\n[preference-bandit 5-seed extension] incompatible completed run will not be reused: %s\n' "$completed"
    fi
    printf '\n[preference-bandit 5-seed extension] start seed %s: %s\n' "$seed" "$output"
    "$PY" scripts/train_preference_bandit.py \
      --seed "$seed" --max_episodes 50 --feature-mode hybrid6_raw \
      --history-ablation full --reward-mode rd_scan \
      --lowfreq-evidence-path "$EV" --epsilon_end 0.05 --per_alpha 0.6 \
      --per_beta_frames auto --gradient_clip 50.0 --output_dir "$output"
    completed="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
    test -n "$completed"
    validate_run "$completed"
  fi
  NEW_RUNS+=("$completed")
done

comparison="outputs/preference_bandit_${RUN_ID}_5seed_extension"
"$PY" scripts/analyze_preference_bandit.py \
  --shared-runs "${REFERENCE_RUNS[@]}" "${NEW_RUNS[@]}" \
  --expected-seeds 42,45,46,47,48 \
  --output "$comparison"
test -s "$comparison/preference_bandit_summary.csv"
printf '\n[preference-conditioned contextual bandit 5-seed extension] completed: %s\n' "$comparison"
