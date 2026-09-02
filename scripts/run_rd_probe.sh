#!/usr/bin/env bash
# Recovery–Discovery 权重探针（2026-09-02）：固定当前已修复 RL 机制，不改 reward 结构，
# 只换 reward = w_recovery×recovery(legacy) + w_discovery×discovery(evidenceV2 低频新候选) 的权重。
#
# 设计（用户裁定）：
#   - 5 组权重 (w_rec, w_disc)：(1,0) / (0.8,0.2) / (0.5,0.5) / (0.2,0.8) / (0,1)
#   - 每组 3 个 seed（42/43/44，各组共用，便于配对比较），每轮只跑 10 episodes。
#   - 每轮按 Validation NDCG 保存 best checkpoint（既有逻辑，不动）。
#   - 评估输出四指标：Recovery→NDCG@150/Recall@150；Discovery→LowFreqNovel@150/EvidenceSupportedLowFreqNovel@150
#   - 判定：若权重升高能稳定、同向改变这些指标 → 进 preference-conditioned MORL；
#           否则暂停 MORL，重新定义 Recovery/Discovery。
#
# 机制参数与修复后完全一致：PER alpha 0.6 / beta auto / grad clip 50 / epsilon_end 0.05 /
# train_label_bonus 1.0 / 不新增任何 Train-driver-进-Top-K 类奖励。
#
# 产物：
#   outputs/rdprobe_r{rec}_d{disc}_seed{42..44}/hybrid6_raw/.../  训练 run（每轮 train_metrics.csv）
#   outputs/rdprobe_eval/summary_metrics.csv                      四指标（含 Discovery 两列）
#   outputs/rdprobe_eval/rankings/*.csv                           排名文件
#   outputs/rdprobe_eval/rd_probe_group_summary.csv               mean±SD + 方向性判定
#
# 用法（正式训练由用户手动运行，约 3-4 小时，建议后台/过夜）：
#   nohup bash scripts/run_rd_probe.sh > outputs/rdprobe.log 2>&1 &

set -u
cd /mnt/e/Projects/RL-GenRisk-main || exit 1
PY=python3
EV="/mnt/e/codex_file/二阶段/06_低频机制V2正式验证/01_evidence/low_frequency_evidence_table_internal_v2.csv"

run_one() {
  local name="$1"; local wr="$2"; local wd="$3"; local seed="$4"
  echo ""
  echo "======================================================"
  echo "== [$name] w_rec=$wr w_disc=$wd seed=$seed 开始 $(date '+%H:%M:%S')"
  echo "======================================================"
  "$PY" src/train.py \
    --feature-mode hybrid6_raw \
    --max_episodes 10 \
    --seed "$seed" \
    --reward-mode rd_scan \
    --w-recovery "$wr" \
    --w-discovery "$wd" \
    --rd-evidence-min 0.5 \
    --rd-evidence-scale 2.0 \
    --rd-evidence-cap 5.0 \
    --lowfreq-evidence-path "$EV" \
    --epsilon_end 0.05 \
    --per_alpha 0.6 \
    --per_beta_frames auto \
    --gradient_clip 50.0 \
    --output_dir "outputs/$name"
  echo "== [$name] 结束，退出码 $? $(date '+%H:%M:%S')"
}

SEEDS="42 43 44"
for s in $SEEDS; do
  run_one "rdprobe_r100_d000_seed$s" 1.0 0.0 "$s"
  run_one "rdprobe_r080_d020_seed$s" 0.8 0.2 "$s"
  run_one "rdprobe_r050_d050_seed$s" 0.5 0.5 "$s"
  run_one "rdprobe_r020_d080_seed$s" 0.2 0.8 "$s"
  run_one "rdprobe_r000_d100_seed$s" 0.0 1.0 "$s"
done

echo ""
echo "=== 训练完成，统一 greedy-rollout 评估（15 runs）==="
"$PY" scripts/evaluate_greedy_rollout.py \
  --run-dirs "outputs/rdprobe_r*/hybrid6_raw/*" \
  --output outputs/rdprobe_eval

echo ""
echo "=== 权重扫描分析（mean±SD + 方向性判定）==="
"$PY" scripts/analyze_rd_probe.py \
  --eval-dir outputs/rdprobe_eval \
  --output-dir outputs/rdprobe_eval

echo ""
echo "全部结束。看结论："
echo "  汇总表  outputs/rdprobe_eval/rd_probe_group_summary.csv"
echo "  判定    直接看运行日志末尾，或 cat outputs/rdprobe_eval/rd_probe_group_summary.csv"
