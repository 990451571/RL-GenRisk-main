#!/usr/bin/env bash
# 多种子验证：最佳配置(exp6 = 低频 V2 + 去命中) vs 基准(legacy)，各跑 seed 42/43/44 做配对对比。
#
# 用法：
#   bash scripts/run_multiseed_validation.sh
# 后台（推荐）：
#   nohup bash scripts/run_multiseed_validation.sh > outputs/multiseed_validation.log 2>&1 &

set -u
cd /mnt/e/Projects/RL-GenRisk-main || exit 1
PY=python3

V2="/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv"
SCALE=1.2396
CAP=1.0

run_one() {
  local name="$1"; local seed="$2"; shift 2
  echo ""
  echo "======================================================"
  echo "== [$name] seed=$seed 开始 $(date '+%H:%M:%S')"
  echo "======================================================"
  "$PY" src/train.py \
    --feature-mode hybrid6_raw \
    --max_episodes 50 \
    --seed "$seed" \
    "$@" \
    --output_dir "outputs/$name"
  echo "== [$name] 结束，退出码 $? $(date '+%H:%M:%S')"
}

# 基准 legacy × 3 种子
for s in 42 43 44; do
  run_one "exp1_legacy_seed$s" "$s" --reward-mode legacy
done

# 最佳配置 V2 + 去命中 × 3 种子
for s in 42 43 44; do
  run_one "exp6_v2bonus0_seed$s" "$s" \
    --reward-mode lowfreq_unlabeled_evidence_v2 \
    --lowfreq-evidence-path "$V2" \
    --lowfreq-unlabeled-bonus-scale "$SCALE" --lowfreq-unlabeled-bonus-cap "$CAP" \
    --train-label-bonus 0.0
done

echo ""
echo "全部结束。结果在各 outputs/exp*_seed*/ 下；出对比图："
echo "  python scripts/visualize_results.py --run-dirs outputs/exp*_seed*/*/* --output outputs/multiseed_compare.html"
