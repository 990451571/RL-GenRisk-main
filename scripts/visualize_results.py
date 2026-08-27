#!/usr/bin/env python3
"""生成 RL-GenRisk 运行结果的自包含 HTML 可视化报告。

支持两种输入模式：

1. 多种子 × 多配置对比（pilot 结构）：
     <result-dir>/seed_<N>/<config>/{episode_metrics,learn_metrics,actions}.csv、final_summary.json
   用法：--result-dir <dir> [--seeds ...] [--configs ...]

2. 正式训练单个 / 多个 run（src/train.py 产出结构）：
     <run-dir>/train_metrics.csv、validation_metrics_episode_*.json、
                validation_ranking_*.csv、summary.json
   用法：--run-dir <dir>  或  --run-dirs <dir1> <dir2> ...

输出是单个自包含 HTML 文件（数据内嵌），浏览器直接打开即可，无需本地服务器。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# 正式训练 train_metrics.csv 里会被画成「分量均值曲线」的 reward 分量列（见 src/train.py REWARD_COMPONENT_KEYS）
REWARD_COMPONENT_KEYS = [
    "reward_legacy",
    "reward_train_label",
    "reward_mutation",
    "reward_expression",
    "reward_methylation",
    "reward_lowfreq",
    "reward_evidence_bonus",
]
# 与上面的分量一一对应的分类色（参考调色板 slot 1-7）
REWARD_COMPONENT_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str) -> float | None:
    val = row.get(key)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 共享 CSS 与图表 JS（两种模式共用）
# ---------------------------------------------------------------------------

SHARED_CSS = r"""
:root { color-scheme: light; }
.viz-root {
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --border: rgba(11,11,11,0.10);
  color-scheme: light;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 28px 32px 8px; }
header h1 { font-size: 20px; margin: 0 0 6px; font-weight: 600; }
header p { margin: 0; color: var(--ink-2); font-size: 13px; line-height: 1.5; }
.tag { display: inline-block; background: #eef2f7; color: var(--ink-2);
  font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-right: 6px; }
main { padding: 12px 32px 48px; display: grid; gap: 16px; }
.panel { background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 18px 20px 14px; }
.panel h2 { font-size: 14px; margin: 0 0 4px; font-weight: 600; }
.panel .sub { color: var(--muted); font-size: 12px; margin: 0 0 14px; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.chart { position: relative; }
.chart svg { width: 100%; height: auto; display: block; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }
.legend .item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--ink-2); }
.legend .swatch { width: 14px; height: 3px; border-radius: 2px; }
.legend .dot { width: 8px; height: 8px; border-radius: 50%; }
#tooltip { position: fixed; pointer-events: none; display: none; background: var(--ink);
  color: #fff; font-size: 12px; padding: 6px 9px; border-radius: 6px; z-index: 10;
  max-width: 260px; line-height: 1.4; }
#tooltip .trow { display: flex; gap: 8px; justify-content: space-between; }
#tooltip .trow b { font-weight: 600; }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
@media (max-width: 760px) { .row { grid-template-columns: 1fr; } }
"""

SHARED_JS = r"""
const NS = "http://www.w3.org/2000/svg";
const CSS = getComputedStyle(document.body);
const C = (n) => CSS.getPropertyValue(n).trim();
const S1 = C("--s1"), S2 = C("--s2"), S3 = C("--s3"), S4 = C("--s4");
const INK = C("--ink"), INK2 = C("--ink-2"), MUTED = C("--muted"), GRID = C("--grid"), AXIS = C("--axis");
const tip = document.getElementById("tooltip");
function showTip(html, x, y) { tip.innerHTML = html; tip.style.display = "block";
  tip.style.left = (x + 14) + "px"; tip.style.top = (y + 14) + "px"; }
function hideTip() { tip.style.display = "none"; }
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function fmt(v, d) { return (v == null || isNaN(v)) ? "—" : Number(v).toLocaleString("en-US", {maximumFractionDigits: d}); }

function niceTicks(min, max, n) {
  if (min === max) { max = min + 1; min = min - 1; }
  const span = max - min, step0 = span / Math.max(1, n - 1), mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag; let step = mag;
  if (norm < 1.5) step = mag; else if (norm < 3) step = 2 * mag; else if (norm < 7) step = 5 * mag; else step = 10 * mag;
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step, ticks = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) ticks.push(v);
  return ticks;
}

function lineChart(id, opts) {
  const box = document.getElementById(id);
  const W = 640, H = 260, mL = 46, mR = 12, mT = 16, mB = 34;
  const iw = W - mL - mR, ih = H - mT - mB;
  const allX = [], allY = [];
  opts.series.forEach(s => s.points.forEach(p => { allX.push(p[0]); allY.push(p[1]); }));
  const x0 = Math.min(...allX), x1 = Math.max(...allX), y0 = Math.min(...allY), y1 = Math.max(...allY);
  const sx = v => mL + (v - x0) / (x1 - x0 || 1) * iw;
  const sy = v => mT + (1 - (v - y0) / (y1 - y0 || 1)) * ih;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  const leg = document.createElement("div"); leg.className = "legend";
  opts.series.forEach(s => { const it = document.createElement("div"); it.className = "item";
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = s.color;
    const lb = document.createElement("span"); lb.textContent = s.name;
    it.append(sw, lb); leg.appendChild(it); });
  box.prepend(leg);
  niceTicks(y0, y1, 5).forEach(t => {
    const y = sy(t);
    el("line", { x1: mL, x2: W - mR, y1: y, y2: y, stroke: GRID, "stroke-width": 1 }, svg);
    const tx = el("text", { x: mL - 8, y: y + 4, "text-anchor": "end", "font-size": 11, fill: MUTED }, svg);
    tx.textContent = fmt(t, opts.yFmt);
  });
  niceTicks(x0, x1, 5).forEach(t => { const x = sx(t);
    el("line", { x1: x, x2: x, y1: mT, y2: mT + ih, stroke: GRID, "stroke-width": 1 }, svg);
    const tx = el("text", { x: x, y: H - 12, "text-anchor": "middle", "font-size": 11, fill: MUTED }, svg);
    tx.textContent = fmt(t, 0);
  });
  opts.series.forEach(s => {
    const d = s.points.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" }, svg);
    s.points.forEach(p => { el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 3.5, fill: s.color, stroke: CSS.getPropertyValue("--surface-1"), "stroke-width": 2 }, svg); });
  });
  const cross = el("line", { y1: mT, y2: mT + ih, stroke: MUTED, "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0 }, svg);
  const overlay = el("rect", { x: mL, y: mT, width: iw, height: ih, fill: "transparent" }, svg);
  overlay.addEventListener("mousemove", ev => {
    const rect = svg.getBoundingClientRect();
    const px = (ev.clientX - rect.left) / rect.width * W;
    const vx = x0 + (px - mL) / iw * (x1 - x0);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("opacity", 1);
    let html = `<div class="trow"><b>${opts.xLabel || "x"}</b><b>${fmt(vx, 2)}</b></div>`;
    opts.series.forEach(s => {
      let best = s.points[0]; let bd = 1e18;
      s.points.forEach(p => { const d2 = Math.abs(p[0] - vx); if (d2 < bd) { bd = d2; best = p; } });
      html += `<div class="trow"><span>${s.name}</span><span>${fmt(best[1], opts.yFmt)}</span></div>`;
    });
    showTip(html, ev.clientX, ev.clientY);
  });
  overlay.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
}

function hBarChart(id, items, color) {
  const box = document.getElementById(id);
  const W = 640, labelW = 92, mR = 46, rowH = 22, gap = 2, mT = 8, mB = 8;
  const H = mT + mB + items.length * (rowH + gap);
  const maxV = Math.max(...items.map(i => i.value), 1e-9);
  const minV = Math.min(...items.map(i => i.value), 0);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  const x0 = labelW, x1 = W - mR;
  const sx = v => x0 + (v - Math.min(0, minV)) / (Math.max(maxV, Math.abs(minV)) || 1) * (x1 - x0);
  const zero = sx(0);
  items.forEach((it, i) => {
    const y = mT + i * (rowH + gap);
    const g = el("text", { x: labelW - 8, y: y + rowH - 5, "text-anchor": "end", "font-size": 11, fill: INK2 }, svg);
    g.textContent = it.label;
    const bw = Math.abs(sx(it.value) - zero);
    el("rect", { x: Math.min(sx(it.value), zero), y: y, width: Math.max(bw, 0.5), height: rowH - gap, rx: 2, fill: color, opacity: 0.85 }, svg);
    const v = el("text", { x: sx(it.value) + (it.value >= 0 ? 6 : -6), y: y + rowH - 5, "text-anchor": it.value >= 0 ? "start" : "end", "font-size": 10.5, fill: MUTED }, svg);
    v.textContent = fmt(it.value, 4);
  });
  el("line", { x1: zero, x2: zero, y1: mT, y2: H - mB, stroke: AXIS, "stroke-width": 1 }, svg);
}

function boxPlot(id, title, higherBetter, series, pointLabels, pointTerm) {
  const box = document.getElementById(id);
  const W = 640, H = 250, mL = 46, mR = 12, mT = 16, mB = 34;
  const iw = W - mL - mR, ih = H - mT - mB;
  const all = series.flatMap(s => s.values);
  const y0 = Math.min(...all), y1 = Math.max(...all);
  const pad = (y1 - y0 || 1) * 0.15;
  const sy = v => mT + (1 - (v - (y0 - pad)) / ((y1 + pad) - (y0 - pad))) * ih;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  const leg = document.createElement("div"); leg.className = "legend";
  series.forEach(s => { const it = document.createElement("div"); it.className = "item";
    const d = document.createElement("span"); d.className = "dot"; d.style.background = s.color;
    const lb = document.createElement("span"); lb.textContent = s.name;
    it.append(d, lb); leg.appendChild(it); });
  box.prepend(leg);
  const titleEl = document.createElement("div"); titleEl.className = "sub";
  titleEl.style.margin = "0 0 6px"; titleEl.textContent = title + (higherBetter ? "（越大越好）" : "（越小越好）");
  box.prepend(titleEl);
  niceTicks(y0 - pad, y1 + pad, 5).forEach(t => {
    const y = sy(t);
    el("line", { x1: mL, x2: W - mR, y1: y, y2: y, stroke: GRID, "stroke-width": 1 }, svg);
    const tx = el("text", { x: mL - 8, y: y + 4, "text-anchor": "end", "font-size": 11, fill: MUTED }, svg);
    tx.textContent = fmt(t, 3);
  });
  const nGroups = series.length, band = iw / nGroups, bw = Math.min(band * 0.5, 90);
  series.forEach((s, gi) => {
    const cx = mL + band * gi + band / 2;
    const vals = [...s.values].sort((a, b) => a - b);
    const n = vals.length;
    const q = p => { const i = (n - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return vals[lo] + (vals[hi] - vals[lo]) * (i - lo); };
    const q1 = q(0.25), med = q(0.5), q3 = q(0.75), mn = vals[0], mx = vals[n - 1];
    el("line", { x1: cx, x2: cx, y1: sy(mx), y2: sy(q3), stroke: s.color, "stroke-width": 1.5 }, svg);
    el("line", { x1: cx, x2: cx, y1: sy(mn), y2: sy(q1), stroke: s.color, "stroke-width": 1.5 }, svg);
    el("line", { x1: cx - bw * 0.3, x2: cx + bw * 0.3, y1: sy(mx), y2: sy(mx), stroke: s.color, "stroke-width": 1.5 }, svg);
    el("line", { x1: cx - bw * 0.3, x2: cx + bw * 0.3, y1: sy(mn), y2: sy(mn), stroke: s.color, "stroke-width": 1.5 }, svg);
    const boxRect = el("rect", { x: cx - bw / 2, y: sy(q3), width: bw, height: Math.max(sy(q1) - sy(q3), 0.5), rx: 2, fill: s.color, opacity: 0.18, stroke: s.color, "stroke-width": 1.5 }, svg);
    boxRect.addEventListener("mousemove", ev => {
      showTip(`<div class="trow"><span>${s.name}</span><b>中位数 ${fmt(med, 4)}</b></div><div class="trow"><span>Q1</span><span>${fmt(q1, 4)}</span></div><div class="trow"><span>Q3</span><span>${fmt(q3, 4)}</span></div><div class="trow"><span>min / max</span><span>${fmt(mn, 4)} / ${fmt(mx, 4)}</span></div>`, ev.clientX, ev.clientY);
    });
    boxRect.addEventListener("mouseleave", hideTip);
    el("line", { x1: cx - bw / 2, x2: cx + bw / 2, y1: sy(med), y2: sy(med), stroke: s.color, "stroke-width": 2 }, svg);
    s.values.forEach((v, si) => {
      const jx = cx - bw / 2 + (bw * (si + 0.5)) / n;
      el("circle", { cx: jx, cy: sy(v), r: 4, fill: s.color, stroke: CSS.getPropertyValue("--surface-1"), "stroke-width": 2 }, svg);
      const hit = el("circle", { cx: jx, cy: sy(v), r: 11, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", ev => {
        showTip(`<div class="trow"><span>${s.name}</span><b>${fmt(v, 4)}</b></div><div class="trow"><span>${pointTerm}</span><span>${pointLabels[si]}</span></div>`, ev.clientX, ev.clientY);
      });
      hit.addEventListener("mouseleave", hideTip);
    });
    const gl = el("text", { x: cx, y: H - 12, "text-anchor": "middle", "font-size": 11, fill: INK2 }, svg);
    gl.textContent = s.name;
  });
}
"""


def assemble_html(title: str, body: str, render_js: str, data: dict) -> str:
    return (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title}</title><style>{SHARED_CSS}</style></head>"
        f"<body class='viz-root'>{body}<div id='tooltip'></div>"
        f"<script>const DATA = {json.dumps(data, ensure_ascii=False)};{SHARED_JS}{render_js}</script>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# 模式 1：多种子 × 多配置（pilot 结构）
# ---------------------------------------------------------------------------

def extract_pilot_data(result_dir: Path, seeds: list[int], configs: list[str],
                       rep_seed: int, rep_config: str, topk: int) -> dict:
    box = {cfg: {"final_reward": [], "loss_mean": [], "reward_per_step": []} for cfg in configs}
    for seed in seeds:
        for cfg in configs:
            d = result_dir / f"seed_{seed}" / cfg
            eps_path = d / "episode_metrics.csv"
            lr_path = d / "learn_metrics.csv"
            if not eps_path.exists():
                raise FileNotFoundError(f"缺少 {eps_path}")
            eps = read_csv(eps_path)
            final_reward = _float(eps[-1], "episode_total_reward")
            reward_per_step = sum(_float(r, "mean_reward_per_step") or 0.0 for r in eps) / len(eps)
            loss_mean = None
            if lr_path.exists():
                lr = read_csv(lr_path)
                if lr:
                    loss_mean = sum(_float(r, "loss") or 0.0 for r in lr) / len(lr)
            box[cfg]["final_reward"].append(round(final_reward, 4) if final_reward is not None else None)
            box[cfg]["loss_mean"].append(round(loss_mean, 5) if loss_mean is not None else None)
            box[cfg]["reward_per_step"].append(round(reward_per_step, 5))

    rep = result_dir / f"seed_{rep_seed}" / rep_config
    acts = read_csv(rep / "actions.csv")
    learn = read_csv(rep / "learn_metrics.csv")
    eps = read_csv(rep / "episode_metrics.csv")

    step_n = 60
    idx = [round(i * (len(acts) - 1) / (step_n - 1)) for i in range(step_n)] if len(acts) > step_n else range(len(acts))
    epsilon = {"steps": [int(acts[i]["step"]) for i in idx], "values": [_float(acts[i], "epsilon") for i in idx]}
    learn_curve = {
        "learn_index": [int(r["learn_index"]) for r in learn],
        "loss": [_float(r, "loss") for r in learn],
        "td": [_float(r, "TD_error_abs_mean") for r in learn],
        "grad": [_float(r, "gradient_norm") for r in learn],
    }
    ep_reward = {"episodes": [int(r["episode"]) for r in eps], "values": [_float(r, "episode_total_reward") for r in eps]}
    gene_reward = defaultdict(float)
    for r in acts:
        gene_reward[r["gene"]] += _float(r, "reward") or 0.0
    top_genes = [{"gene": g, "reward": round(rw, 5)} for g, rw in sorted(gene_reward.items(), key=lambda kv: -kv[1])[:topk]]

    return {
        "seeds": seeds, "configs": configs, "rep_seed": rep_seed, "rep_config": rep_config,
        "box": box, "epsilon": epsilon, "learn_curve": learn_curve, "ep_reward": ep_reward, "top_genes": top_genes,
    }


PILOT_BODY = r"""
<header>
  <h1>RL-GenRisk 运行结果可视化 · 对比</h1>
  <p><span class="tag" id="tag-configs"></span><span class="tag" id="tag-seeds"></span></p>
  <p style="margin-top:6px" id="subtitle"></p>
</header>
<main>
  <section class="panel">
    <h2>箱线图 · 配置对比</h2>
    <p class="sub">每个箱子 = 该配置在多个种子下的分布；中位线看「典型水平」，箱子长短看「稳不稳」，散点是每个种子的实际值。</p>
    <div class="row">
      <div class="chart" id="box-final_reward"></div>
      <div class="chart" id="box-loss_mean"></div>
    </div>
    <div class="note">final reward 越大越好；loss 越小越好。</div>
  </section>
  <section class="panel">
    <h2>训练动态 · seed <span id="rep-seed"></span> / <span id="rep-config"></span></h2>
    <p class="sub">每个 learn 步的损失 / TD 误差 / 梯度范数，以及每轮总奖励。</p>
    <div class="row"><div class="chart" id="learn-curve"></div><div class="chart" id="ep-reward"></div></div>
  </section>
  <section class="panel">
    <h2>基因排名 · 累计奖励 Top <span id="topk"></span></h2>
    <p class="sub">按「该基因在整轮中被选中时累计获得的总奖励」排序。</p>
    <div class="chart" id="top-genes"></div>
  </section>
</main>
"""

PILOT_RENDER = r"""
document.getElementById("tag-configs").textContent = "配置：" + DATA.configs.join(" / ");
document.getElementById("tag-seeds").textContent = "种子：" + DATA.seeds.join(" / ");
document.getElementById("subtitle").textContent = "曲线来自代表 run（seed " + DATA.rep_seed + " / " + DATA.rep_config + "）。";
document.getElementById("rep-seed").textContent = DATA.rep_seed;
document.getElementById("rep-config").textContent = DATA.rep_config;
document.getElementById("topk").textContent = DATA.top_genes.length;

const boxDefs = [
  { key: "final_reward", title: "最终 episode 奖励", higherBetter: true },
  { key: "loss_mean", title: "平均 Loss", higherBetter: false },
];
const SERIES = DATA.configs.map((c, i) => ({ name: c, color: [S1, S2, S3][i % 3] }));
boxDefs.forEach(bd => {
  boxPlot("box-" + bd.key, bd.title, bd.higherBetter,
    SERIES.map(s => ({ name: s.name, color: s.color, values: DATA.box[s.name][bd.key] })),
    DATA.seeds, "种子");
});

lineChart("learn-curve", {
  xLabel: "learn 步", yLabel: "数值", yFmt: 2,
  series: [
    { name: "loss", color: S1, points: DATA.learn_curve.learn_index.map((x, i) => [x, DATA.learn_curve.loss[i]]) },
    { name: "TD 误差(绝对值)", color: S2, points: DATA.learn_curve.learn_index.map((x, i) => [x, DATA.learn_curve.td[i]]) },
    { name: "梯度范数", color: S3, points: DATA.learn_curve.learn_index.map((x, i) => [x, DATA.learn_curve.grad[i]]) },
  ],
});
lineChart("ep-reward", {
  xLabel: "episode", yLabel: "总奖励", yFmt: 3,
  series: [ { name: "episode 总奖励", color: S1, points: DATA.ep_reward.episodes.map((x, i) => [x, DATA.ep_reward.values[i]]) } ],
});
hBarChart("top-genes", DATA.top_genes.map(g => ({ label: g.gene, value: g.reward })), S1);
"""


# ---------------------------------------------------------------------------
# 模式 2：正式训练 run（src/train.py 产出结构）
# ---------------------------------------------------------------------------

def _read_formal_run(run_dir: Path, topk: int) -> dict:
    tm = read_csv(run_dir / "train_metrics.csv")
    episodes = [int(r["episode"]) for r in tm]
    components = {key: [_float(r, key + "_mean") for r in tm] for key in REWARD_COMPONENT_KEYS}

    val_eps, ndcg, prec, rec, mrr = [], [], [], [], []
    for vf in sorted(run_dir.glob("validation_metrics_episode_*.json")):
        m = json.loads(vf.read_text(encoding="utf-8"))
        val_eps.append(int(m["episode"]))
        ndcg.append(m.get("NDCG@150"))
        prec.append(m.get("Precision@150"))
        rec.append(m.get("Recall@150"))
        mrr.append(m.get("MRR"))

    summary = {}
    sf = run_dir / "summary.json"
    if sf.exists():
        summary = json.loads(sf.read_text(encoding="utf-8"))

    top_genes = []
    rank_file = run_dir / "validation_ranking_best.csv"
    if not rank_file.exists():
        rank_files = sorted(run_dir.glob("validation_ranking_episode_*.csv"))
        rank_file = rank_files[-1] if rank_files else None
    if rank_file is not None and rank_file.exists():
        rows = read_csv(rank_file)
        top_genes = [{"gene": r["Gene"], "q": float(r["Q_value"])} for r in rows[:topk]]

    best_ndcg = summary.get("best_val_ndcg150")
    if best_ndcg is None:
        best_ndcg = ndcg[-1] if ndcg else None

    return {
        "name": run_dir.name,
        "episodes": episodes,
        "loss": [_float(r, "mean_loss") for r in tm],
        "td": [_float(r, "td_error_abs_mean") for r in tm],
        "grad": [_float(r, "gradient_norm_mean") for r in tm],
        "epsilon": [_float(r, "epsilon") for r in tm],
        "reward": [_float(r, "episode_reward") for r in tm],
        "components": components,
        "val_episodes": val_eps, "ndcg150": ndcg, "prec150": prec, "rec150": rec, "mrr": mrr,
        "best_ndcg": best_ndcg,
        "best_episode": summary.get("best_episode"),
        "top_genes": top_genes,
    }


def extract_formal_data(run_dirs: list[Path], topk: int) -> dict:
    runs = [_read_formal_run(d, topk) for d in run_dirs]
    return {
        "run_names": [r["name"] for r in runs],
        "best_ndcg": [r["best_ndcg"] for r in runs],
        "rep": runs[0],
        "n_runs": len(runs),
    }


FORMAL_BODY = r"""
<header>
  <h1>RL-GenRisk 训练结果可视化</h1>
  <p><span class="tag" id="tag-runs"></span></p>
  <p style="margin-top:6px" id="subtitle"></p>
</header>
<main>
  <section class="panel">
    <h2>最佳 NDCG@150 · 跨 run</h2>
    <p class="sub">每个点是一个 run 的最佳验证 NDCG@150；多个 run 时箱子显示分布（越大越好）。</p>
    <div class="chart" id="box-best-ndcg"></div>
  </section>
  <section class="panel">
    <h2>训练曲线 · <span id="rep-name"></span></h2>
    <p class="sub">损失 / TD 误差 / 梯度范数 / 探索率 / 每轮奖励随轮次变化。</p>
    <div class="row"><div class="chart" id="train-loss"></div><div class="chart" id="train-reward"></div></div>
    <div class="row" style="margin-top:16px"><div class="chart" id="train-epsilon"></div><div class="chart" id="train-grad"></div></div>
  </section>
  <section class="panel">
    <h2>验证曲线 · <span id="rep-name2"></span></h2>
    <p class="sub">NDCG@150 / Precision@150 / Recall@150 / MRR 随轮次变化。</p>
    <div class="row"><div class="chart" id="val-ndcg"></div><div class="chart" id="val-mrr"></div></div>
  </section>
  <section class="panel">
    <h2>Reward 分解</h2>
    <p class="sub">各 reward 分量的均值随轮次变化。</p>
    <div class="chart" id="reward-components"></div>
  </section>
  <section class="panel">
    <h2>基因排名 · 按 Q 值 Top <span id="topk"></span></h2>
    <p class="sub">来自最终排名（validation_ranking_best.csv）。</p>
    <div class="chart" id="top-genes"></div>
  </section>
</main>
"""

FORMAL_RENDER = r"""
document.getElementById("tag-runs").textContent = "run 数：" + DATA.n_runs;
document.getElementById("subtitle").textContent = "曲线来自代表 run：" + DATA.rep.name;
document.getElementById("rep-name").textContent = DATA.rep.name;
document.getElementById("rep-name2").textContent = DATA.rep.name;
document.getElementById("topk").textContent = DATA.rep.top_genes.length;

boxPlot("box-best-ndcg", "最佳 NDCG@150", true,
  [{ name: "NDCG@150", color: S1, values: DATA.best_ndcg }], DATA.run_names, "run");

const R = DATA.rep;
lineChart("train-loss", { xLabel: "episode", yLabel: "mean_loss", yFmt: 4,
  series: [ { name: "mean_loss", color: S1, points: R.episodes.map((x, i) => [x, R.loss[i]]) } ] });
lineChart("train-reward", { xLabel: "episode", yLabel: "总奖励", yFmt: 3,
  series: [ { name: "episode_reward", color: S1, points: R.episodes.map((x, i) => [x, R.reward[i]]) } ] });
lineChart("train-epsilon", { xLabel: "episode", yLabel: "epsilon", yFmt: 3,
  series: [ { name: "epsilon", color: S1, points: R.episodes.map((x, i) => [x, R.epsilon[i]]) } ] });
lineChart("train-grad", { xLabel: "episode", yLabel: "数值", yFmt: 2,
  series: [
    { name: "梯度范数", color: S1, points: R.episodes.map((x, i) => [x, R.grad[i]]) },
    { name: "TD 误差(绝对值)", color: S2, points: R.episodes.map((x, i) => [x, R.td[i]]) },
  ] });

lineChart("val-ndcg", { xLabel: "episode", yLabel: "指标", yFmt: 4,
  series: [
    { name: "NDCG@150", color: S1, points: R.val_episodes.map((x, i) => [x, R.ndcg150[i]]) },
    { name: "Precision@150", color: S2, points: R.val_episodes.map((x, i) => [x, R.prec150[i]]) },
    { name: "Recall@150", color: S3, points: R.val_episodes.map((x, i) => [x, R.rec150[i]]) },
  ] });
lineChart("val-mrr", { xLabel: "episode", yLabel: "MRR", yFmt: 4,
  series: [ { name: "MRR", color: S1, points: R.val_episodes.map((x, i) => [x, R.mrr[i]]) } ] });

const compKeys = __COMP_KEYS__;
const compColors = __COMP_COLORS__;
lineChart("reward-components", { xLabel: "episode", yLabel: "分量均值", yFmt: 4,
  series: compKeys.map((k, i) => ({ name: k.replace("reward_", ""), color: compColors[i],
    points: R.episodes.map((x, j) => [x, R.components[k][j]]) })) });

hBarChart("top-genes", R.top_genes.map(g => ({ label: g.gene, value: g.q })), S1);
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 RL-GenRisk 结果可视化 HTML")
    parser.add_argument("--result-dir", default=None, help="[模式1] 多种子×多配置结果目录")
    parser.add_argument("--run-dir", default=None, help="[模式2] 单个正式训练 run 目录")
    parser.add_argument("--run-dirs", nargs="+", default=None, help="[模式2] 多个正式训练 run 目录")
    parser.add_argument("--output", default=None, help="输出 HTML 路径")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="[模式1] 种子列表")
    parser.add_argument("--configs", nargs="+", default=["original", "multiomics"], help="[模式1] 配置列表")
    parser.add_argument("--rep-seed", type=int, default=None, help="[模式1] 代表 run 的种子")
    parser.add_argument("--rep-config", default=None, help="[模式1] 代表 run 的配置")
    parser.add_argument("--topk", type=int, default=20, help="基因排名条数")
    args = parser.parse_args()

    modes = [args.result_dir, args.run_dir, args.run_dirs]
    if sum(1 for m in modes if m) != 1:
        parser.error("请且仅指定一种输入：--result-dir / --run-dir / --run-dirs 之一")

    out = Path(args.output) if args.output else Path("outputs/visualization_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.result_dir:
        result_dir = Path(args.result_dir)
        rep_seed = args.rep_seed if args.rep_seed is not None else args.seeds[0]
        rep_config = args.rep_config if args.rep_config is not None else args.configs[0]
        if rep_config not in args.configs:
            raise ValueError(f"--rep-config {rep_config} 不在 --configs 中")
        data = extract_pilot_data(result_dir, args.seeds, args.configs, rep_seed, rep_config, args.topk)
        html = assemble_html("RL-GenRisk 运行结果可视化 · 对比", PILOT_BODY, PILOT_RENDER, data)
    else:
        run_dirs = [Path(args.run_dir)] if args.run_dir else [Path(d) for d in args.run_dirs]
        for d in run_dirs:
            if not (d / "train_metrics.csv").exists():
                raise FileNotFoundError(f"{d} 不是正式训练 run 目录（缺少 train_metrics.csv）")
        data = extract_formal_data(run_dirs, args.topk)
        render = FORMAL_RENDER.replace("__COMP_KEYS__", json.dumps(REWARD_COMPONENT_KEYS)).replace(
            "__COMP_COLORS__", json.dumps(REWARD_COMPONENT_COLORS))
        html = assemble_html("RL-GenRisk 训练结果可视化", FORMAL_BODY, render, data)

    out.write_text(html, encoding="utf-8")
    print(f"已生成可视化报告：{out}")


if __name__ == "__main__":
    main()
