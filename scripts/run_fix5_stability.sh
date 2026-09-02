#!/usr/bin/env bash
# 修复第二阶段（2026-09-02）：当前修复机制原样不动，补跑 5 个新 seed，
# 验证「前 1-3 轮最好、后续下降」是否稳定现象 + 修复效果在多 seed 下是否可重复。
#
# 与第一阶段完全相同的机制参数（reward 已修 / PER alpha 0.6 / beta auto / clip 50 / epsilon 0.05）。
# 不加任何新模块、不调学习内容；只换 seed 45-49。禁止新增“Train driver 进 Top-K”类奖励。
#
# 产物：
#   outputs/fix_legacy_seed45..49/        —— 每个含 train_metrics.csv（每轮 reward/NDCG/Recall）
#   outputs/fix_all_rollout_eval/          —— 8 个 seed(42-49)统一 greedy-rollout 评估
#        ├ summary_metrics.csv             （pretrain/best/last 三时点 NDCG/Recall/Hits）
#        ├ stability_curves.csv            （每 seed 曲线：峰值轮、跌幅、reward 是否走高）
#        └ pre_best_last_compare.csv       （三时点 Top-150 重合 + Spearman）
#
# 用法（用户手动运行正式训练，5 种子 × ~53 分钟 + 评估 ≈ 5 小时，建议后台/过夜）：
#   nohup bash scripts/run_fix5_stability.sh > outputs/fix5_stability.log 2>&1 &

set -u
cd /mnt/e/Projects/RL-GenRisk-main || exit 1
PY=python3

run_one() {
  local name="$1"; local seed="$2"
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

# 新 seed 45-49（与 42/43/44 同机制）
for s in 45 46 47 48 49; do
  run_one "fix_legacy_seed$s" "$s"
done

echo ""
echo "=== 训练完成，统一 greedy-rollout 评估（42-49 共 8 个 seed）==="
"$PY" scripts/evaluate_greedy_rollout.py \
  --run-dirs "outputs/fix_legacy_seed4*/hybrid6_raw/*" \
  --output outputs/fix_all_rollout_eval

echo ""
echo "=== 稳定性与三时点分析（8 个 seed）==="
"$PY" scripts/analyze_stability_fix.py \
  --metrics-runs "outputs/fix_legacy_seed4{2,3,4,5,6,7,8,9}/hybrid6_raw/*" \
  --eval-dir outputs/fix_all_rollout_eval \
  --output-dir outputs/fix_all_rollout_eval

echo ""
echo "全部结束。看结论："
echo "  python3 -c \"import pandas as pd; print(pd.read_csv('outputs/fix_all_rollout_eval/pre_best_last_compare.csv').to_string(index=False))\""
echo "  或直接看运行日志中的『稳定性判定』小节。"
