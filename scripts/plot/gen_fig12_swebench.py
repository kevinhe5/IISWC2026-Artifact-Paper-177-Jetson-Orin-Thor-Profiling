#!/usr/bin/env python3
"""Fig 12 — SWE-bench-live agentic trace stats: Dense (Llama-3.2-1B) vs
reasoning (Qwen3-4B), 4 frameworks per regime (TRT-LLM excluded — no
reasoning counterpart on sm_87). 2×3 grid.

Formatting preserved from the paper's original generator
(JetsonAnalysis/figs/scripts/gen_fig_act2_cdf_combined_from_csv.py.bak):
same panel layout, same CDF step style, same median markers + summary
box, same 8-row action-mix bar with framework-tinted y-ticks, same
top-level fw/regime legend.

Data intake replaced: reads per-turn traces from
    data/agentic/llama_1B/{fw}.csv           (dense: Llama-3.2-1B)
    data/agentic/qwen3_4B/{fw}_thinkON.csv   (reason: Qwen3-4B, think-ON)
The Qwen3 llamacpp file may be either `_thinkON` or plain, so we try
both. Paper filename: fig_act2_cdf_combined.pdf.

  python3 gen_fig12_swebench.py [--out DIR]
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "agentic"

FW_LIST   = ["vllm", "sglang", "llamacpp", "pytorch"]
REGIMES   = ["dense", "reason"]
FW_COLORS = {"vllm": "#f97316", "sglang": "#a78bfa",
             "llamacpp": "#f472b6", "pytorch": "#34d399"}
STYLE = {"dense": (0, (4, 2.5)), "reason": "-"}

ACTION_ORDER = ["respond", "list_dir", "view_file", "grep", "edit_file",
                "bash", "pytest", "run_python", "done"]
ACTION_COLOR = {
    "respond":        "#d4d4d8",
    "list_dir":       "#fde68a",
    "view_file":      "#BDD4E5",
    "grep":           "#fbcfe8",
    "edit_file":      "#1f4f9c",
    "bash":           "#3a8b3a",
    "pytest":         "#dc2626",
    "run_python":     "#9333ea",
    "done":           "#f59e0b",
    "parser_garbage": "#7f1d1d",
    "other":          "#a3a3a3",
}
KNOWN = set(ACTION_ORDER)


def csv_for(fw, regime):
    if regime == "dense":
        return DATA / "llama_1B" / f"{fw}.csv"
    # reason: prefer explicit _thinkON, fall back to plain filename
    for name in (f"{fw}_thinkON.csv", f"{fw}.csv"):
        p = DATA / "qwen3_4B" / name
        if p.exists():
            return p
    return DATA / "qwen3_4B" / f"{fw}_thinkON.csv"   # will fail loudly


def load_one(fw, regime):
    p = csv_for(fw, regime)
    rows = []
    if not p.exists():
        return rows, p.name
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                ms = float(r.get("phase_ms") or 0)
                ex = json.loads(r["extra"]) if r.get("extra") else {}
            except Exception:
                continue
            try:
                turn = int(r.get("turn_idx") or 0)
            except ValueError:
                continue
            rows.append({
                "task_id":  r.get("task_id", ""),
                "turn_idx": turn,
                "phase":    r.get("phase", ""),
                "phase_ms": ms,
                "extra":    ex if isinstance(ex, dict) else {},
            })
    return rows, p.name


def cdf(values):
    if not values:
        return [], []
    xs = sorted(values)
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    return xs, ys


def plot_cdf_panel(ax, data, title, xlabel, *, log_x=True, fmt="{:.0f}"):
    medians = {}
    for (fw, regime), vals in data.items():
        if not vals:
            continue
        xs, ys = cdf(vals)
        ax.step(xs, ys, where="post",
                color=FW_COLORS[fw],
                linestyle=STYLE[regime],
                linewidth=1.7 if regime == "reason" else 1.4,
                alpha=0.95, zorder=3)
        medians[(fw, regime)] = median(vals)

    ax.axhline(0.5, color="#9ca3af", linewidth=0.6, linestyle=":",
               alpha=0.7, zorder=2)

    for (fw, regime), m in medians.items():
        ax.scatter(m, 0.5, s=90, color=FW_COLORS[fw],
                   edgecolor="#0f1115", linewidth=0.8,
                   marker="o",
                   facecolor=FW_COLORS[fw] if regime == "reason" else "white",
                   zorder=5)

    lines = []
    for fw in FW_LIST:
        d = medians.get((fw, "dense"))
        r = medians.get((fw, "reason"))
        if d is None or r is None:
            continue
        ratio = (r / d) if d else 0
        lines.append((fw, d, r, ratio))

    if lines:
        text_lines = ["med (NR → R)"]
        for fw, d, r, ratio in lines:
            text_lines.append(
                f"{fw:8s} {fmt.format(d):>7s} → {fmt.format(r):>7s}  "
                f"({ratio:.1f}×)"
            )
        ax.text(0.02, 0.98, "\n".join(text_lines),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.2, fontfamily="monospace",
                color="#1f2937",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#d1d5db", lw=0.6, alpha=0.92),
                zorder=6)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title(title, fontsize=10)
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _bucket(a):
    if not a or not isinstance(a, str):
        return "other"
    for known in ACTION_ORDER:
        if a != known and a.startswith(known) and len(a) > len(known):
            return "parser_garbage"
    return a if a in KNOWN else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    rows_by = {}
    for fw in FW_LIST:
        for regime in REGIMES:
            rs, fn = load_one(fw, regime)
            rows_by[(fw, regime)] = rs
            print(f"  {fw:9s} {regime:6s} {len(rs):5d} rows  <- {fn}")

    # Paper layout: 1x3 — (a) iteration depth | (b) response length | (c) action mix
    fig, (axA, axC, axF) = plt.subplots(
        1, 3, figsize=(19.5, 4.9),
        gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.55], wspace=0.24))

    # (a) Iteration depth
    depth = {}
    for (fw, regime), rows in rows_by.items():
        by_task = defaultdict(set)
        for r in rows:
            by_task[r["task_id"]].add(r["turn_idx"])
        depth[(fw, regime)] = [len(ts) for ts in by_task.values()]
    plot_cdf_panel(axA, depth,
                   title="(a) Iteration depth", xlabel="turns per task",
                   log_x=False, fmt="{:.0f}")

    # (b) Response length per turn
    gen = {}
    for (fw, regime), rows in rows_by.items():
        gen[(fw, regime)] = [
            r["extra"].get("gen_tokens", 0) for r in rows
            if r["phase"] == "decode" and (r["extra"].get("gen_tokens") or 0) > 0
        ]
    plot_cdf_panel(axC, gen,
                   title="(b) Response length per turn",
                   xlabel="gen tokens (log)", log_x=True)

    # (c) Action mix — 8-row paired horizontal bars; non-reasoning rows hatched
    row_keys = []
    for fw in FW_LIST:
        row_keys.append((fw, "reason"))
        row_keys.append((fw, "dense"))

    stack_order = ACTION_ORDER + ["parser_garbage", "other"]
    y_pos = np.arange(len(row_keys))[::-1]

    for i, key in enumerate(row_keys):
        fw, regime = key
        cnt = defaultdict(int)
        for r in rows_by[key]:
            if r["phase"] != "parse_action": continue
            cnt[_bucket(r["extra"].get("action"))] += 1
        tot = sum(cnt.values()) or 1
        pct = {k: 100 * v / tot for k, v in cnt.items()}
        left = 0
        for stk in stack_order:
            v = pct.get(stk, 0.0)
            if v <= 0: continue
            axF.barh(y_pos[i], v, height=0.78,
                     left=left, color=ACTION_COLOR[stk],
                     edgecolor="#0f1115", linewidth=0.4, zorder=3,
                     hatch="///" if regime == "dense" else None)
            if v >= 7.0:
                axF.text(left + v / 2, y_pos[i], f"{v:.0f}",
                         ha="center", va="center", fontsize=7.0,
                         color=("white" if stk in (
                             "edit_file", "bash", "pytest", "run_python",
                             "parser_garbage") else "#1f2937"),
                         fontweight="bold", zorder=4)
            left += v

    ylabels = [f"{fw}  ·  {'reason' if regime == 'reason' else 'non-reason'}"
               for fw, regime in row_keys]
    axF.set_yticks(y_pos)
    axF.set_yticklabels(ylabels, fontsize=8.5)
    for tick_label, (fw, _) in zip(axF.get_yticklabels(), row_keys):
        tick_label.set_color(FW_COLORS[fw])
        tick_label.set_fontweight("bold")
    axF.tick_params(axis="y", length=0)
    axF.set_xlim(0, 100)
    axF.set_xticks([0, 25, 50, 75, 100])
    axF.set_xticklabels(["0", "25", "50", "75", "100 %"], fontsize=8.5)
    axF.set_xlabel("share of actions emitted by the model", fontsize=10)
    axF.set_title("(c) Action mix: non-reasoning vs reasoning per framework",
                  fontsize=10)
    axF.grid(True, axis="x", color="#e5e7eb", linewidth=0.4, alpha=0.7, zorder=1)
    for s in ("top", "right"):
        axF.spines[s].set_visible(False)
    for j in range(1, len(FW_LIST)):
        sep_y = y_pos[2 * j] + 0.5
        axF.axhline(sep_y, color="#d4d4d8", linewidth=0.4, linestyle=":",
                    alpha=0.7, zorder=1)

    # Action legend (below panel c)
    from matplotlib.patches import Patch
    used = set()
    for key in row_keys:
        for r in rows_by[key]:
            if r["phase"] == "parse_action":
                used.add(_bucket(r["extra"].get("action")))
    legend_keys = [k for k in stack_order if k in used]
    handles_act = [Patch(facecolor=ACTION_COLOR[k], edgecolor="#0f1115",
                         linewidth=0.4, label=k) for k in legend_keys]
    axF.legend(handles=handles_act, loc="upper center",
               bbox_to_anchor=(0.5, -0.20),
               ncol=min(len(handles_act), 5), frameon=False, fontsize=7.5)

    # Top-level fw + regime legend
    from matplotlib.lines import Line2D
    fw_handles = [Line2D([0], [0], color=FW_COLORS[fw], lw=2.4, label=fw)
                  for fw in FW_LIST]
    regime_handles = [
        Line2D([0], [0], color="#1f2937", lw=1.8, linestyle="-",
               label="reasoning  (Qwen3-4B, solid)"),
        Line2D([0], [0], color="#1f2937", lw=1.5, linestyle=(0, (4, 2.5)),
               label="non-reasoning  (Llama-3.2-1B, dashed)"),
    ]
    fig.legend(handles=fw_handles + regime_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.06),
               ncol=6, frameon=False, fontsize=11)

    fig.tight_layout(pad=1.2, w_pad=1.6, rect=(0, 0, 1, 0.96))
    pdf = out_dir / "fig12_swebench.pdf"
    png = out_dir / "fig12_swebench.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05, dpi=180)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
