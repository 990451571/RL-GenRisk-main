#!/usr/bin/env bash
# 低频奖励实验：7 个 run 顺序执行
# 对比维度：reward 模式（legacy / lowfreq V1 / lowfreq V2）× train-label-bonus（默认 1.0 / 0.0 / 0.5）
#
# 用法（前台）：
#   bash scripts/run_lowfreq_experiments.sh
# 用法（后台，适合长时间跑，断开终端不中断）：
#   nohup bash scripts/run_lowfreq_experiments.sh > outputs/lowfreq_experiments.log 2>&1 &
#
# 每个 run 输出到 outputs/<name>/ 下，train.py 会自动在其内创建带时间戳的子目录。

set -u  # 未定义变量即报错；不 set -e，让单个 run 失败后继续跑下一个

cd /mnt/e/Projects/RL-GenRisk-main || exit 1
PY=python3

V1="/mnt/e/codex_file/二阶段/05_低频癌基因发现机制/01_evidence/low_frequency_evidence_table.csv"
V2="/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv"
SCALE=1.2396
CAP=1.0

run_one() {
  local name="$1"; shift
  echo ""
  echo "======================================================"
  echo "== [$name] 开始  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "======================================================"
  "$PY" src/train.py \
    --feature-mode hybrid6_raw \
    --seed 0 \
    --max_episodes 50 \
    "$@" \
    --output_dir "outputs/$name"
  echo "== [$name] 结束，退出码 $?  $(date '+%Y-%m-%d %H:%M:%S')"
}

# 1) 基准对照：legacy（train-label-bonus 默认 1.0）
run_one exp1_baseline_legacy \
  --reward-mode legacy

# 2) 低频 V1，默认命中奖励
run_one exp2_lowfreq_v1_default \
  --reward-mode lowfreq_unlabeled_evidence \
  --lowfreq-evidence-path "$V1" \
  --lowfreq-unlabeled-bonus-scale "$SCALE" --lowfreq-unlabeled-bonus-cap "$CAP"

# 3) 低频 V1，去掉直接命中奖励
run_one exp3_lowfreq_v1_bonus0 \
  --reward-mode lowfreq_unlabeled_evidence \
  --lowfreq-evidence-path "$V1" \
  --lowfreq-unlabeled-bonus-scale "$SCALE" --lowfreq-unlabeled-bonus-cap "$CAP" \
  --train-label-bonus 0.0

# 4) 低频 V1，命中奖励减半
run_one exp4_lowfreq_v1_bonus0p5 \
  --reward-mode lowfreq_unlabeled_evidence \
  --lowfreq-evidence-path "$V1" \
  --lowfreq-unlabeled-bonus-scale "$SCALE" --lowfreq-unlabeled-bonus-cap "$CAP" \
  --train-label-bonus 0.5

# 5) 低频 V2，默认命中奖励
run_one exp5_lowfreq_v2_default \
  --reward-mode lowfreq_unlabeled_evidence_v2 \
  --lowfreq-evidence-path "$V2" \
  --lowfreq-unlabeled-bonus-scale "$SCALE" --lowfreq-unlabeled-bonus-cap "$CAP"

# 6) 低频 V2，去掉直接命中奖励
run_one exp6_lowfreq_v2_bonus0 \
  --reward-mode lowfreq_unlabeled_evidence_v2 \
  --lowfreq-evidence-path "$V2" \
  --lowfreq-unlabeled-bonus-scale "$SCALE" --lowfreq-unlabeled-bonus-cap "$CAP" \
  --train-label-bonus 0.0

# 7) 低频 V2，命中奖励减半
run_one exp7_lowfreq_v2_bonus0p5 \
  --reward-mode lowfreq_unlabeled_evidence_v2 \
  --lowfreq-evidence-path "$V2" \
  --lowfreq-unlabeled-bonus-scale "$SCALE" --lowfreq-unlabeled-bonus-cap "$CAP" \
  --train-label-bonus 0.5

echo ""
echo "全部结束。各 run 结果在 outputs/exp*/ 下；可用以下命令出图对比："
echo "  python scripts/visualize_results.py --run-dirs outputs/exp*/*/* --output outputs/lowfreq_compare.html"
