#!/usr/bin/env bash
# RL 学习回路修复后的 legacy/recovery 目标验证（2026-09-01）。
#
# 目的：只修已确认的机制问题，不新增模块，验证「训练后」是否真的比「训练前」好。
#   - reward/10 已移除（learn() 中不再除 10.0）
#   - PER: alpha 0.2→0.6，beta 退火周期自动匹配训练长度（不再卡在 0.1）
#   - 梯度裁剪 1.0→50.0（不再全程强裁）
#   - epsilon 下限 0.15→0.05（减少每轮随机动作）
# 以上均已作为 src/train.py 的新默认值，本脚本显式传入以保证可复现。
#
# 对比口径（训练前 vs 训练后）：
#   evaluate_greedy_rollout.py 会输出 pretrain_Q_onepass / best_Q_onepass 等排名，
#   summary_metrics.csv 给出 NDCG@150 / Recall@150 / HitCount@150；
#   Top-K overlap 与 rank correlation 由 scripts/compare_prepost_fix.py 计算。
#
# 用法（用户手动运行正式训练，约 50 分钟/种子 ×3）：
#   nohup bash scripts/run_mechanism_fix_validation.sh > outputs/fix_validation.log 2>&1 &

set -u
cd /mnt/e/Projects/RL-GenRisk-main || exit 1
PY=python3

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
    --reward-mode legacy \
    --epsilon_end 0.05 \
    --per_alpha 0.6 \
    --per_beta_frames auto \
    --gradient_clip 50.0 \
    --output_dir "outputs/$name"
  echo "== [$name] 结束，退出码 $? $(date '+%H:%M:%S')"
}

# 修复后的 legacy × 3 种子（与旧 exp1_legacy_seed* 同结构，便于直接对比）
for s in 42 43 44; do
  run_one "fix_legacy_seed$s" "$s"
done

echo ""
echo "=== 训练完成，开始训练前/后评估（greedy rollout）==="
"$PY" scripts/evaluate_greedy_rollout.py \
  --run-dirs "outputs/fix_legacy_seed*/hybrid6_raw/*" \
  --output outputs/fix_rollout_eval

echo ""
echo "=== 计算训练前/后 Top-K overlap 与 rank correlation ==="
"$PY" scripts/compare_prepost_fix.py \
  --eval-dir outputs/fix_rollout_eval \
  --output outputs/fix_rollout_eval/prepost_compare.csv

echo ""
echo "全部结束。结果："
echo "  指标        outputs/fix_rollout_eval/summary_metrics.csv"
echo "  前后对比    outputs/fix_rollout_eval/prepost_compare.csv"
echo "  排名文件    outputs/fix_rollout_eval/rankings/*.csv"
echo "看关键结论：cat outputs/fix_rollout_eval/prepost_compare.csv"
