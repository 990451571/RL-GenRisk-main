#!/usr/bin/env bash
# Validation-only DDQN history ablation.  Full, no-history and shuffled-history
# share all training settings; only the Q-visible selected-gene context differs.
set -eu
cd /mnt/e/Projects/RL-GenRisk-main

PY="${PY:-python3}"
export PYTHONUNBUFFERED=1
EV=/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUNS_FILE=""

run_one() {
  local mode="$1" tag="$2" wr="$3" wd="$4" seed="$5"
  local output="outputs/historyablation_${RUN_ID}_${mode}_${tag}_seed${seed}"
  local completed=""
  if [ -d "$output/hybrid6_raw" ]; then
    completed="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
  fi
  if [ -n "$completed" ] && "$PY" -c 'import json, sys; a=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(not (a.get("learning_mode") == "ddqn" and a.get("history_ablation") == sys.argv[2]))' "$completed/config.json" "$mode"; then
    printf '\n[%s] reuse completed run: %s\n' "$mode" "$completed"
    printf '%s\n' "$completed" >> "$RUNS_FILE"
    return
  fi
  if [ -n "$completed" ]; then
    printf '\n[%s] ignore incompatible completed run: %s\n' "$mode" "$completed"
  fi
  printf '\n[%s] seed=%s preference=%s output=%s\n' "$mode" "$seed" "$tag" "$output"
  "$PY" src/train.py \
    --learning-mode ddqn --history-ablation "$mode" \
    --feature-mode hybrid6_raw --max_episodes 10 --seed "$seed" \
    --reward-mode rd_scan --w-recovery "$wr" --w-discovery "$wd" \
    --rd-evidence-min 0.5 --rd-evidence-scale 2.0 --rd-evidence-cap 5.0 \
    --lowfreq-evidence-path "$EV" --epsilon_end 0.05 --per_alpha 0.6 \
    --per_beta_frames auto --gradient_clip 50.0 --output_dir "$output"
  completed="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
  test -n "$completed"
  "$PY" -c 'import json, sys; a=json.load(open(sys.argv[1], encoding="utf-8")); assert a.get("learning_mode") == "ddqn" and a.get("history_ablation") == sys.argv[2]' "$completed/config.json" "$mode"
  printf '%s\n' "$completed" >> "$RUNS_FILE"
}

run_mode() {
  local mode="$1"
  RUNS_FILE="/tmp/rlgenrisk_history_${RUN_ID}_${mode}.paths"
  : > "$RUNS_FILE"
  for seed in 42 45 46; do
    run_one "$mode" r100_d000 1.0 0.0 "$seed"
    run_one "$mode" r080_d020 0.8 0.2 "$seed"
    run_one "$mode" r050_d050 0.5 0.5 "$seed"
    run_one "$mode" r020_d080 0.2 0.8 "$seed"
    run_one "$mode" r000_d100 0.0 1.0 "$seed"
  done
  test "$(wc -l < "$RUNS_FILE")" -eq 15
}

for mode in full no_history shuffled_history; do
  run_mode "$mode"
done

FULL_EVAL="outputs/historyablation_${RUN_ID}_full_eval"
NO_HISTORY_EVAL="outputs/historyablation_${RUN_ID}_no_history_eval"
SHUFFLED_EVAL="outputs/historyablation_${RUN_ID}_shuffled_history_eval"

mapfile -t FULL_RUNS < "/tmp/rlgenrisk_history_${RUN_ID}_full.paths"
mapfile -t NO_HISTORY_RUNS < "/tmp/rlgenrisk_history_${RUN_ID}_no_history.paths"
mapfile -t SHUFFLED_RUNS < "/tmp/rlgenrisk_history_${RUN_ID}_shuffled_history.paths"
"$PY" scripts/evaluate_greedy_rollout.py --run-dirs "${FULL_RUNS[@]}" --output "$FULL_EVAL"
"$PY" scripts/evaluate_greedy_rollout.py --run-dirs "${NO_HISTORY_RUNS[@]}" --output "$NO_HISTORY_EVAL"
"$PY" scripts/evaluate_greedy_rollout.py --run-dirs "${SHUFFLED_RUNS[@]}" --output "$SHUFFLED_EVAL"

COMPARE="outputs/historyablation_${RUN_ID}_comparison"
"$PY" scripts/analyze_history_ablation.py \
  --full-eval "$FULL_EVAL" \
  --no-history-eval "$NO_HISTORY_EVAL" \
  --shuffled-history-eval "$SHUFFLED_EVAL" \
  --output "$COMPARE"
test -s "$COMPARE/history_ablation_summary.csv"
printf '\n[DDQN history ablation] completed: %s\n' "$COMPARE"
