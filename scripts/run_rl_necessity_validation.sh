#!/usr/bin/env bash
# Fair RL-necessity validation: existing scalar DDQN versus contextual bandit.
# Both use the same rd_scan scalar reward, features, labels, seeds, rollout
# evaluation and ten-episode exposure.  The bandit alone removes bootstrap.
set -eu
cd /mnt/e/Projects/RL-GenRisk-main

PY="${PY:-python3}"
export PYTHONUNBUFFERED=1
EV=/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv
# Set RUN_ID explicitly to resume an interrupted validation.  A completed
# bandit run is identified by its summary.json and is never trained again.
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BANDIT_RUNS=()
DDQN_RUNS=()

run_one() {
  local tag="$1" wr="$2" wd="$3" seed="$4"
  local output="outputs/rlnecessity_bandit_${RUN_ID}_${tag}_seed${seed}"
  local existing=""
  if [ -d "$output/hybrid6_raw" ]; then
    existing="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' -exec test -f '{}/summary.json' ';' -print | sort | tail -n 1)"
  fi
  if [ -n "$existing" ] && "$PY" -c 'import json, sys; sys.exit(json.load(open(sys.argv[1], encoding="utf-8")).get("learning_mode") != "contextual_bandit")' "$existing/config.json"; then
    printf '\n[bandit] reuse completed run: %s\n' "$existing"
    BANDIT_RUNS+=("$existing")
    return
  fi
  if [ -n "$existing" ]; then
    printf '\n[bandit] ignore incompatible completed run (not contextual_bandit): %s\n' "$existing"
  fi
  printf '\n[bandit] seed=%s preference=%s output=%s\n' "$seed" "$tag" "$output"
  "$PY" src/train.py \
    --learning-mode contextual_bandit \
    --feature-mode hybrid6_raw --max_episodes 10 --seed "$seed" \
    --reward-mode rd_scan --w-recovery "$wr" --w-discovery "$wd" \
    --rd-evidence-min 0.5 --rd-evidence-scale 2.0 --rd-evidence-cap 5.0 \
    --lowfreq-evidence-path "$EV" --epsilon_end 0.05 --per_alpha 0.6 \
    --per_beta_frames auto --gradient_clip 50.0 --output_dir "$output"
  local run
  run="$(find "$output/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | sort | tail -n 1)"
  test -n "$run" && test -f "$run/summary.json"
  BANDIT_RUNS+=("$run")
}

for seed in 42 45 46; do
  run_one r100_d000 1.0 0.0 "$seed"
  run_one r080_d020 0.8 0.2 "$seed"
  run_one r050_d050 0.5 0.5 "$seed"
  run_one r020_d080 0.2 0.8 "$seed"
  run_one r000_d100 0.0 1.0 "$seed"
  for tag in r100_d000 r080_d020 r050_d050 r020_d080 r000_d100; do
    reference="$(find "outputs/rdprobe_${tag}_seed${seed}/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | sort | tail -n 1)"
    test -n "$reference" && test -f "$reference/config.json"
    DDQN_RUNS+=("$reference")
  done
done

EVAL_DIR="outputs/rlnecessity_bandit_${RUN_ID}_eval"
"$PY" scripts/evaluate_greedy_rollout.py --run-dirs "${BANDIT_RUNS[@]}" --output "$EVAL_DIR"
test -s "$EVAL_DIR/summary_metrics.csv"
DDQN_EVAL_DIR="outputs/rlnecessity_ddqn_${RUN_ID}_eval"
"$PY" scripts/evaluate_greedy_rollout.py --run-dirs "${DDQN_RUNS[@]}" --output "$DDQN_EVAL_DIR"
test -s "$DDQN_EVAL_DIR/summary_metrics.csv"

# Supervised MLP/GCN use only Train-16 labels.  They are ranking controls, not
# reward-trained agents; each is seeded independently for honest stability.
SUP_ROOT="outputs/rlnecessity_supervised_${RUN_ID}"
SUP_DIRS=()
for seed in 42 45 46; do
  reference="$(find "outputs/rdprobe_r100_d000_seed${seed}/hybrid6_raw" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | sort | tail -n 1)"
  test -n "$reference" && test -f "$reference/config.json"
  out="$SUP_ROOT/seed${seed}"
  if [ -s "$out/supervised_summary.csv" ]; then
    printf '\n[supervised] reuse completed result: %s\n' "$out"
  else
    "$PY" scripts/train_supervised_baseline.py --run-dir "$reference" --output "$out" --epochs 200
  fi
  test -s "$out/supervised_summary.csv"
  SUP_DIRS+=("$out")
done

COMPARE="outputs/rlnecessity_${RUN_ID}_comparison"
"$PY" scripts/analyze_rl_necessity.py \
  --ddqn-eval "$DDQN_EVAL_DIR" \
  --bandit-eval "$EVAL_DIR" \
  --supervised-dirs "${SUP_DIRS[@]}" \
  --output "$COMPARE"
test -s "$COMPARE/rl_necessity_summary.csv"
printf '\n[RL necessity] completed: %s\n' "$COMPARE"
