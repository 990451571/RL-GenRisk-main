#!/usr/bin/env bash
# Shared preference-conditioned contextual bandit, validation only.
# 50 episodes = each of the five frozen preferences is trained exactly 10 times.
set -eu
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${PY:-python3}"
export PYTHONUNBUFFERED=1
EV=/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIRS=()

for seed in 42 45 46; do
  output="outputs/preference_bandit_${RUN_ID}_seed${seed}"
  completed=""
  if [ -d "$output/hybrid6_raw" ]; then
    completed="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
  fi
  if [ -n "$completed" ] && "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); assert d["status"] == "COMPLETED"; c=json.load(open(sys.argv[2],encoding="utf-8")); assert c["learning_mode"] == "contextual_bandit" and c["td_target"] == "immediate_reward_only" and c["schedule_policy"] == "balanced_latin_blocks_v1" and set(c["trained_preference_counts"].values()) == {10} and all(v == [2,2,2,2,2] for v in c["trained_preference_phase_counts"].values())' "$completed/summary.json" "$completed/preference_bandit_config.json"; then
    printf '\n[preference-bandit] reuse seed %s: %s\n' "$seed" "$completed"
  else
    if [ -n "$completed" ]; then
      printf '\n[preference-bandit] incompatible completed run will not be reused: %s\n' "$completed"
    fi
    printf '\n[preference-bandit] start seed %s/46: %s\n' "$seed" "$output"
    "$PY" scripts/train_preference_bandit.py \
      --seed "$seed" --max_episodes 50 --feature-mode hybrid6_raw \
      --history-ablation full --reward-mode rd_scan \
      --lowfreq-evidence-path "$EV" --epsilon_end 0.05 --per_alpha 0.6 \
      --per_beta_frames auto --gradient_clip 50.0 --output_dir "$output"
    completed="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
    test -n "$completed"
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); c=json.load(open(sys.argv[2],encoding="utf-8")); assert d["status"] == "COMPLETED"; assert c["learning_mode"] == "contextual_bandit" and c["td_target"] == "immediate_reward_only" and c["schedule_policy"] == "balanced_latin_blocks_v1" and set(c["trained_preference_counts"].values()) == {10} and all(v == [2,2,2,2,2] for v in c["trained_preference_phase_counts"].values())' "$completed/summary.json" "$completed/preference_bandit_config.json"
  fi
  RUN_DIRS+=("$completed")
done

comparison="outputs/preference_bandit_${RUN_ID}_comparison"
"$PY" scripts/analyze_preference_bandit.py \
  --shared-runs "${RUN_DIRS[@]}" \
  --ddqn-eval outputs/rlnecessity_ddqn_20260907_152513_eval \
  --scalar-bandit-eval outputs/rlnecessity_bandit_20260907_152513_eval \
  --supervised-dirs outputs/rlnecessity_supervised_20260907_152513/seed42 outputs/rlnecessity_supervised_20260907_152513/seed45 outputs/rlnecessity_supervised_20260907_152513/seed46 \
  --output "$comparison"
test -s "$comparison/preference_bandit_summary.csv"
printf '\n[preference-conditioned contextual bandit] completed: %s\n' "$comparison"
