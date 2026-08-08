#!/usr/bin/env python3
"""Fig 11 — Pareto: decode throughput vs energy efficiency, Llama-3.2-1B pp=128/gen=128.
AGX Orin (hollow markers) vs AGX Thor (filled). De-hardcoded: both platforms read shipped CSVs.

  Orin: data/chat/sweep_locked.csv  -> x = decode_tps ; y = decode_tps / P_decode(4-rail)
  Thor: data/chat/pareto_thor/pareto_thor_base.csv (tps, tok_per_j; quant points graphs-ON,
        fp16 eager per the paper's convention — see METHODOLOGY_NOTES).

  python3 gen_fig11_pareto.py [--out DIR]
"""
import argparse, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
ORIN_CSV = REPO / "data/chat/sweep_locked.csv"   # 15-run sweep (raw or means)
ORIN_PC = REPO / "data/chat/pytorch_compile.csv"
THOR_CSV = REPO / "data/chat/pareto_thor/pareto_thor_base.csv"
FP16 = {"fp16", "bf16", "f16", "16-bit"}   # exclude fp16_nocache/mb32 ablations
NAME = {"llamacpp": "llama.cpp", "llamacpp_fa": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang",
        "trtllm": "TRT", "trtedge_llm": "TRT", "pytorch": "PyTorch", "pytorch_compile": "PyTorch+compile"}
COL = {"TRT": "#1f77b4", "llama.cpp": "#ff7f0e", "vLLM": "#9467bd", "SGLang": "#7f7f7f",
       "PyTorch": "#d62728", "PyTorch+compile": "#2ca02c"}
MK = {"TRT": "o", "llama.cpp": "D", "vLLM": "s", "SGLang": "^", "PyTorch": "v", "PyTorch+compile": "P"}
GEN = 128


def orin_pts():
    # average per (framework, quant) — robust to raw N-rows/cell OR 1-row/cell
    import collections
    from statistics import fmean
    acc = collections.OrderedDict()
    for r in csv.DictReader(open(ORIN_CSV)):
        if r["prompt_tokens"] != "128" or r["gen_tokens"] != "128":
            continue
        if (r.get("model") or "Llama-3.2-1B") != "Llama-3.2-1B":
            continue
        q = r["quantization"]
        if q == "fp16_nocache" or q == "fp16_mb32":
            continue
        try:
            tps = float(r["decode_tps"]); pw = (float(r["dec_total_mw"]) + float(r.get("dec_dram_mw") or 0)) / 1000.0
        except (TypeError, ValueError):
            continue
        if tps <= 0 or pw <= 0:
            continue
        acc.setdefault((r["framework"], q), []).append((tps, tps / pw))
    return [(fmean(t for t, _ in v), fmean(j for _, j in v), NAME.get(fw, fw))
            for (fw, q), v in acc.items()]


def thor_pts():
    pts = []
    for r in csv.DictReader(open(THOR_CSV)):
        if r["tps"] in ("", "NA"):
            continue
        pts.append((float(r["tps"]), float(r["tok_per_j"]), NAME.get(r["framework"], r["framework"])))
    return pts


def scatter(ax, pts, filled):
    by = {}
    for x, y, fw in pts:
        by.setdefault(fw, []).append((x, y))
    for fw, ps in by.items():
        ax.scatter([p[0] for p in ps], [p[1] for p in ps], marker=MK.get(fw, "o"), s=95, zorder=4,
                   facecolor=(COL.get(fw, "#888") if filled else "none"),
                   edgecolor=COL.get(fw, "#888"), linewidth=1.6, alpha=0.95)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    scatter(ax, orin_pts(), filled=False)
    scatter(ax, thor_pts(), filled=True)
    ax.set_xlabel("Decode throughput (tokens / s)", fontsize=14)
    ax.set_ylabel("Energy efficiency (tokens / J)", fontsize=14)
    ax.grid(True, color="#e5e7eb", lw=0.6, alpha=0.7, zorder=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fw_leg = [Line2D([], [], marker=MK[f], color="w", markerfacecolor=COL[f], markeredgecolor=COL[f],
                     markersize=9, label=f) for f in COL]
    plat_leg = [Line2D([], [], marker="o", color="w", markerfacecolor="none", markeredgecolor="#333",
                       markersize=9, label="AGX Orin (hollow)"),
                Line2D([], [], marker="o", color="w", markerfacecolor="#333", markeredgecolor="#333",
                       markersize=9, label="AGX Thor (filled)")]
    l1 = ax.legend(handles=fw_leg, loc="upper right", fontsize=10, frameon=True, title="framework")
    ax.add_artist(l1)
    ax.legend(handles=plat_leg, loc="lower right", fontsize=10, frameon=True)
    fig.tight_layout()
    out = Path(a.out) / "fig11_pareto"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}.pdf / .png  (orin {len(orin_pts())} pts, thor {len(thor_pts())} pts)")


if __name__ == "__main__":
    main()
