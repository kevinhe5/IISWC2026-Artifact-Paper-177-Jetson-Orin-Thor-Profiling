#!/usr/bin/env python3
"""Fig 3 — 16-bit decode Memory-Bandwidth Utilization at pp=512 / gen=256, Llama-3.2-1B.
AGX Orin 32 GB (204.8 GB/s) vs AGX Thor 128 GB (273 GB/s), locked clocks.
De-hardcoded: reads per-(platform, framework) TPOT from data/chat/mbu_pp512_gen256.csv;
computes MBU analytically. The model-byte and peak-BW constants are hardware/model spec.

  MBU = (weights + KV(T)) / TPOT / peak_BW,
  weights(fp16)=1.235B*2=2.470 GB ; KV=2*16*8*64*2*T ; T = pp + gen/2 = 640.

  python3 gen_fig03_mbu.py [--out DIR]
"""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "chat" / "mbu_pp512_gen256.csv"

WEIGHTS = 1.235e9 * 2
KV = 2 * 16 * 8 * 64 * 2 * (512 + 256 // 2)
STREAM = WEIGHTS + KV
PEAK = {"orin": 204.8, "thor": 273.0}
COLOR = {"TRT": "#60a5fa", "PyTorch (compile)": "#b5835a", "vLLM": "#f97316",
         "llama.cpp": "#f472b6", "SGLang": "#a78bfa", "PyTorch": "#34d399"}
ORDER = ["TRT", "PyTorch (compile)", "vLLM", "llama.cpp", "SGLang", "PyTorch"]


def mbu(tpot_ms, peak):
    return 100.0 * (STREAM / (tpot_ms / 1000.0)) / (peak * 1e9)


def load():
    tp = {"orin": {}, "thor": {}}
    for r in csv.DictReader(open(SRC)):
        tp[r["platform"]][r["framework"]] = float(r["tpot_ms"])
    return tp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    tp = load()
    fws = [f for f in ORDER if f in tp["orin"] or f in tp["thor"]]
    x = np.arange(len(fws)); bw = 0.40
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for off, plat, hatch in [(-bw / 2, "orin", None), (bw / 2, "thor", "////")]:
        for i, f in enumerate(fws):
            if f not in tp[plat]:
                continue
            m = mbu(tp[plat][f], PEAK[plat]); t = tp[plat][f]
            ax.bar(x[i] + off, m, bw, color=COLOR.get(f, "#888"), edgecolor="#0f1115",
                   linewidth=0.8, hatch=hatch, zorder=3)
            ax.text(x[i] + off, m + 1.2, f"{m:.0f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
            ax.text(x[i] + off, m * 0.55, f"{t:.1f} ms", ha="center", va="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(fws, fontsize=13)
    ax.set_ylim(0, 100); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("% of platform BW peak", fontsize=14)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.7, alpha=0.7, zorder=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = [Patch(facecolor="#60a5fa", edgecolor="#0f1115", label="AGX Orin 32 GB (204.8 GB/s)"),
           Patch(facecolor="#60a5fa", edgecolor="#0f1115", hatch="////", label="AGX Thor 128 GB (273 GB/s)")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=13, frameon=False)
    fig.tight_layout()
    out = Path(a.out) / "fig03_mbu"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print("wrote", out, "| Thor MBU:", {f: round(mbu(tp['thor'][f], PEAK['thor']), 1) for f in fws if f in tp['thor']})


if __name__ == "__main__":
    main()
