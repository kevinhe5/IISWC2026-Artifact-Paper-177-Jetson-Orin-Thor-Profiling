#!/usr/bin/env python3
"""Generate fig_v5_agentic_trace_stats.{pdf,png} — Figure (new).

Mirror of Sutradhara (arXiv 2601.12967) Figure 3, but on the edge AGX
Orin instead of an H200 datacenter trace, using Plan A SWE-bench-live
measurements (N=30 SWE-smith bug instances spanning 21 base repos,
12c-unpinned baseline, max_turns=30, multi-tool + thought-prompting
harness — dispatched 2026-05-15).

Panels:
  (a) Iteration Depth CDF      — turns per task per framework (bimodal:
                                  early-done vs MAX_TURNS=30 cap)
  (b) Tool Fan-Out CDF         — tool calls per iteration (multi-tool
                                  harness — most turns still emit 1
                                  action; ~0.3% emit >1)
  (c) Prompt Length CDF        — intermediate vs final turn
  (d) Response Length CDF      — gen_tokens per turn, intermediate vs final
  (e) Tool Time Ratio CDF      — tool_exec_ms / (prefill+decode+tool_exec)
  (f) Tool Latency Variation   — box plots, normalised to per-tool median

Provenance: sweep_results/<fw>_swebench_live_12c_*.csv (Plan A
unpinned-12c baseline, locked clocks, MAXN, 5 frameworks).
(slim_cpp — an exploratory llama.cpp fork — was dropped from the final paper.)
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import argparse
_AP = argparse.ArgumentParser()
_AP.add_argument("--out", default=".")
_ARGS, _ = _AP.parse_known_args()
REPO = Path(__file__).resolve().parents[2]
# De-hardcoded: read the shipped agentic SWE-bench-live CSVs (dense Llama-3.2-1B, Orin-backed).
SWEEP = REPO / "data" / "agentic" / "llama_1B"
_OUT = Path(_ARGS.out) / "fig12_swebench"
OUT_PDF = str(_OUT.with_suffix(".pdf"))
OUT_PNG = str(_OUT.with_suffix(".png"))

CSVS = {
    "llamacpp":  "llamacpp.csv",
    "vllm":      "vllm.csv",
    "sglang":    "sglang.csv",
    "pytorch":   "pytorch.csv",
    "trtllm":    "trtllm.csv",
}
FW_COLORS = {
    "llamacpp": "#f472b6",
    "vllm":     "#f97316", "sglang":   "#a78bfa",
    "pytorch":  "#34d399", "trtllm":   "#60a5fa",
}


def load_one(pattern):
    """Return list of (task_id, turn_idx, phase, phase_ms, extra) rows."""
    rows = []
    for path in sorted(glob.glob(str(SWEEP / pattern))):
        try:
            with open(path) as f:
                for r in csv.DictReader(f):
                    try:
                        phase_ms = float(r.get("phase_ms") or 0)
                        ex = json.loads(r["extra"]) if r.get("extra") else {}
                    except Exception:
                        continue
                    rows.append({
                        "task_id":  int(r.get("task_id") or 0),
                        "turn_idx": int(r.get("turn_idx") or 0),
                        "phase":    r.get("phase", ""),
                        "phase_ms": phase_ms,
                        "extra":    ex,
                    })
        except Exception:
            pass
    return rows


def cdf(values):
    if not values:
        return [], []
    xs = sorted(values)
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    return xs, ys


def main() -> int:
    fw_rows = {fw: load_one(pat) for fw, pat in CSVS.items()}
    # Quick sanity
    for fw, rows in fw_rows.items():
        print(f"  {fw}: {len(rows)} phase-rows")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    (axA, axB, axE), (axC, axD, axF) = axes

    # ----- (a) Iteration depth: turns per (fw, task) -----
    # The bench emits a terminal-action row at turn_idx = MAX_TURNS - 1 even
    # for tasks that ended early via `done`, so max(turn_idx)+1 always equals
    # MAX_TURNS. The honest depth is the COUNT of unique turn_idx values.
    depths_per_fw = defaultdict(list)
    for fw, rows in fw_rows.items():
        by_task = defaultdict(set)
        for r in rows:
            by_task[r["task_id"]].add(r["turn_idx"])
        for t, turns in by_task.items():
            depths_per_fw[fw].append(len(turns))

    for fw, depths in depths_per_fw.items():
        xs, ys = cdf(depths)
        if xs:
            axA.step(xs, ys, where="post", color=FW_COLORS[fw], lw=1.6,
                     label=f"{fw} (n={len(depths)})")
    axA.set_xlabel("turns per task", fontsize=10)
    axA.set_ylabel("CDF", fontsize=10)
    all_depths = [d for ds in depths_per_fw.values() for d in ds]
    axA.set_title(f"(a) Iteration depth  ·  median {median(all_depths):.0f} turns, max_turns=30",
                  fontsize=10)
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=8, loc="lower right", framealpha=0.85)

    # ----- (b) Tool fan-out: tool calls per iteration -----
    # Our bench: single-action ReAct → fan-out = 1 per turn (always one tool_exec phase per turn).
    # We plot the contrast against Sutradhara's multi-tool patterns.
    fanouts = []
    for fw, rows in fw_rows.items():
        per_turn = defaultdict(int)
        for r in rows:
            if r["phase"] == "tool_exec":
                per_turn[(fw, r["task_id"], r["turn_idx"])] += 1
        fanouts.extend(per_turn.values())
    xs, ys = cdf(fanouts)
    axB.step(xs, ys, where="post", color="#f97316", lw=2.0,
             label=f"Edge SWE-bench (n={len(fanouts)})")
    axB.set_xlabel("tool calls per iteration", fontsize=10)
    axB.set_ylabel("CDF", fontsize=10)
    axB.set_title("(b) Tool fan-out  ·  ~100% emit 1 call (multi-tool harness, 1B model)",
                  fontsize=10)
    axB.set_xlim(0, 5)
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=8, loc="center right", framealpha=0.85)

    # ----- (c) Prompt length CDF: intermediate vs final turn -----
    intermediate, final = [], []
    for fw, rows in fw_rows.items():
        by_task = defaultdict(list)
        for r in rows:
            if r["phase"] == "prefill":
                pt = r["extra"].get("prompt_tokens", 0)
                if pt > 0:
                    by_task[r["task_id"]].append((r["turn_idx"], pt))
        for task, turns in by_task.items():
            turns.sort()
            for i, (ti, pt) in enumerate(turns):
                if i == len(turns) - 1:
                    final.append(pt)
                else:
                    intermediate.append(pt)
    xs, ys = cdf(intermediate)
    axC.step(xs, ys, where="post", color="#3b82f6", lw=1.8, ls="--",
             label=f"Intermediate (n={len(intermediate)})")
    xs, ys = cdf(final)
    axC.step(xs, ys, where="post", color="#ef4444", lw=1.8,
             label=f"Final (n={len(final)})")
    axC.set_xlabel("prompt length (tokens)", fontsize=10)
    axC.set_ylabel("CDF", fontsize=10)
    pp_med = median(intermediate) if intermediate else 0
    axC.set_title(f"(c) Prompt length  ·  intermediate-turn median {pp_med:.0f} tok",
                  fontsize=10)
    axC.grid(True, alpha=0.3)
    axC.legend(fontsize=8, loc="lower right", framealpha=0.85)

    # ----- (d) Response length CDF -----
    intermediate, final = [], []
    for fw, rows in fw_rows.items():
        by_task = defaultdict(list)
        for r in rows:
            if r["phase"] == "decode":
                gt = r["extra"].get("gen_tokens", 0)
                if gt > 0:
                    by_task[r["task_id"]].append((r["turn_idx"], gt))
        for task, turns in by_task.items():
            turns.sort()
            for i, (ti, gt) in enumerate(turns):
                if i == len(turns) - 1:
                    final.append(gt)
                else:
                    intermediate.append(gt)
    xs, ys = cdf(intermediate)
    axD.step(xs, ys, where="post", color="#3b82f6", lw=1.8, ls="--",
             label=f"Intermediate (n={len(intermediate)})")
    xs, ys = cdf(final)
    axD.step(xs, ys, where="post", color="#ef4444", lw=1.8,
             label=f"Final (n={len(final)})")
    axD.set_xlabel("response length (gen tokens)", fontsize=10)
    axD.set_ylabel("CDF", fontsize=10)
    gen_med = median(intermediate) if intermediate else 0
    axD.set_title(f"(d) Response length  ·  intermediate-turn median {gen_med:.0f} tok",
                  fontsize=10)
    axD.grid(True, alpha=0.3)
    axD.legend(fontsize=8, loc="lower right", framealpha=0.85)

    # ----- (e) Tool Time Ratio CDF — tool_exec / (prefill+decode+tool_exec) per task -----
    ratios_per_fw = defaultdict(list)
    for fw, rows in fw_rows.items():
        by_task = defaultdict(lambda: {"prefill": 0.0, "decode": 0.0, "tool_exec": 0.0})
        for r in rows:
            if r["phase"] in by_task[0]:
                pass  # noop
            by_task[r["task_id"]].setdefault(r["phase"], 0.0)
            if r["phase"] in ("prefill", "decode", "tool_exec"):
                by_task[r["task_id"]][r["phase"]] = by_task[r["task_id"]].get(r["phase"], 0.0) + r["phase_ms"]
        for t, sums in by_task.items():
            tot = sums.get("prefill", 0) + sums.get("decode", 0) + sums.get("tool_exec", 0)
            if tot > 0:
                ratios_per_fw[fw].append(sums.get("tool_exec", 0) / tot)
    for fw, rs in ratios_per_fw.items():
        xs, ys = cdf(rs)
        if xs:
            axE.step(xs, ys, where="post", color=FW_COLORS[fw], lw=1.6,
                     label=f"{fw} (med {median(rs)*100:.1f}%)")
    axE.set_xlabel("tool / (LLM + tool) per task", fontsize=10)
    axE.set_ylabel("CDF", fontsize=10)
    # Compute fw with the per-task max ratio across all tasks (for the title)
    max_fw, max_r = None, 0.0
    for fw, rs in ratios_per_fw.items():
        if rs and max(rs) > max_r:
            max_fw, max_r = fw, max(rs)
    axE.set_title(f"(e) Tool time share  ·  per-task max {max_r*100:.0f}% on {max_fw}",
                  fontsize=10)
    axE.set_xlim(0, 1.0)
    axE.grid(True, alpha=0.3)
    axE.legend(fontsize=7.5, loc="lower right", framealpha=0.85)

    # ----- (f) Tool latency — horizontal boxplot, absolute ms (log x), per tool -----
    # Exclude the terminal "respond" / "done" actions: synthesized by the
    # bench harness (final-message + termination), no real tool subprocess work.
    EXCLUDE_TOOLS = {"respond", "done"}
    times_per_tool = defaultdict(list)
    for fw, rows in fw_rows.items():
        for r in rows:
            if r["phase"] == "tool_exec":
                tool = r["extra"].get("action") or "?"
                if tool in EXCLUDE_TOOLS:
                    continue
                if r["phase_ms"] > 0:
                    times_per_tool[tool].append(r["phase_ms"])
    # Sort tools by median latency ASCENDING — lightest at top (after invert).
    # Require ≥20 samples so single-call tools (view_file) don't show as a dot.
    tools_sorted = [(t, ts) for t, ts in times_per_tool.items() if len(ts) >= 20]
    tools_sorted.sort(key=lambda kv: median(kv[1]))

    positions = list(range(len(tools_sorted)))
    box_data = [ts for _, ts in tools_sorted]
    yticklabels = [f"{t} (n={len(ts)})" for t, ts in tools_sorted]

    # Light colour for short tools, red for heavy — use a Reds gradient on log(median)
    import matplotlib.colors as mcolors
    meds = [median(ts) for _, ts in tools_sorted]
    norm = mcolors.LogNorm(vmin=max(min(meds), 1e-3), vmax=max(meds))
    cmap = plt.get_cmap("YlOrRd")
    box_colors = [cmap(0.30 + 0.55 * norm(m)) for m in meds]

    bp = axF.boxplot(
        box_data, positions=positions, vert=False, widths=0.65,
        patch_artist=True, showfliers=False, whis=(5, 95),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(color="#52525b", linewidth=0.9),
        capprops=dict(color="#52525b", linewidth=0.9),
        medianprops=dict(color="#7f1d1d", linewidth=1.8),
    )
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set(facecolor=c, edgecolor="#52525b")

    axF.set_yticks(positions)
    axF.set_yticklabels(yticklabels, fontsize=9)
    axF.invert_yaxis()                       # lightest tool at the top
    axF.set_xscale("log")
    axF.set_xlim(0.01, 50000)
    axF.set_xlabel("tool latency (ms, log scale; whiskers = p5–p95)", fontsize=10)
    axF.set_title("(f) Tool latency  ·  4 orders of magnitude (bash → pytest)",
                  fontsize=10)
    axF.grid(True, axis='x', alpha=0.3, which='both')
    axF.tick_params(axis='x', labelsize=9)

    # Inline median annotations to the right of each box
    for i, (_, ts) in enumerate(tools_sorted):
        m = median(ts)
        label = f"{m:.2f} ms" if m < 1 else (f"{m:.1f} ms" if m < 10 else f"{m:.0f} ms")
        # place at p95 (right end of whisker)
        p95 = float(np.percentile(ts, 95))
        axF.text(p95 * 1.6, i, label, va='center', ha='left',
                 fontsize=8.5, color="#7f1d1d", fontweight='bold')

    # Polish: spines + suptitle
    for ax in (axA, axB, axC, axD, axE, axF):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    fig.suptitle(
        "SWE-bench-B agentic trace statistics (Plan A)\n"
        "AGX Orin 32GB · locked clocks · 5 frameworks × 30 SWE-smith bug instances across 21 base repos · max_turns=30 · multi-tool harness",
        fontsize=11.5, y=1.005
    )
    fig.tight_layout(pad=1.4, h_pad=1.6, w_pad=1.4, rect=(0, 0, 1, 0.965))
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.05, dpi=180)
    print(f"\nwrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
