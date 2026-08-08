#!/usr/bin/env python3
"""Fig 4 — Per-token decode latency decomposition from Nsight Systems (AGX Orin | AGX Thor).
De-hardcoded: reads the shipped breakdown JSONs; no embedded DATA.

Per framework the decode wall-time-per-token is split into:
  kernel   = GPU compute (CUPTI HW timestamps)
  launch   = cudaLaunchKernel + graph_launch (CPU launch cost)
  residual = max(wall - kernel - launch, 0)   # exposed sync + memcpy_api + host/Python
Bars sum to wall by construction. SGLang is omitted (its scheduler runs in a spawned
worker nsys cannot follow; see the JSON _note).

  python3 gen_fig04_nsys_breakdown.py [--out DIR]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
NSYS = REPO / "data" / "nsys"
SRC = {"orin": NSYS / "breakdown.json", "thor": NSYS / "breakdown_thor.json"}
TITLE = {"orin": "AGX Orin 32 GB", "thor": "AGX Thor 128 GB"}
ORDER = ["trtllm", "trtedge_llm", "vllm", "llamacpp", "pytorch"]
LABEL = {"trtllm": "TRT-LLM", "trtedge_llm": "TRT-Edge", "vllm": "vLLM",
         "llamacpp": "llama.cpp", "pytorch": "PyTorch"}
COLORS = {"kernel": "#BDD4E5", "launch": "#F7CB9F", "residual": "#E8B4B8"}


def load(plat):
    """-> [(fw, kernel, launch, residual, decode_isolated)] in display order."""
    d = json.load(open(SRC[plat]))
    fws = d["frameworks"] if isinstance(d, dict) else d          # Thor wraps in "frameworks"
    by = {}
    for r in fws:
        fw = r.get("framework")
        if not fw or fw == "sglang":
            continue
        wall = r.get("wall_ms_per_tok", 0.0)
        kernel = r.get("kernel_ms_per_tok", 0.0)
        launch = r.get("launch_ms_per_tok", 0.0) + r.get("graph_launch_ms_per_tok", 0.0)
        # residual defined so bars sum to wall by construction (original Fig 4 convention):
        # residual lumps exposed sync + memcpy_api + other_api + host, minus GPU-overlap.
        # NOT the JSON's residual_ms_per_tok (that is overlap-inclusive and exceeds wall-k-l).
        resid = max(wall - kernel - launch, 0.0)
        iso = r.get("decode_only_via_capture_range", r.get("decode_only_via_nvtx", True))
        by[fw] = (kernel, launch, resid, iso)
    return [(fw, *by[fw]) for fw in ORDER if fw in by]


def panel(ax, plat):
    rows = load(plat)
    xs = np.arange(len(rows)); w = 0.6
    ker = np.array([r[1] for r in rows]); lau = np.array([r[2] for r in rows]); res = np.array([r[3] for r in rows])
    ax.bar(xs, ker, w, color=COLORS["kernel"], label="GPU kernel (compute)")
    ax.bar(xs, lau, w, bottom=ker, color=COLORS["launch"], label="launch (CPU)")
    ax.bar(xs, res, w, bottom=ker + lau, color=COLORS["residual"], label="residual (sync+memcpy+host)")
    for i, r in enumerate(rows):
        wall = r[1] + r[2] + r[3]
        ax.text(i, wall, f"{wall:.1f}", ha="center", va="bottom", fontsize=8)
        if not r[4]:  # not decode-isolated (e.g. TRT-Edge via llm_bench)
            ax.text(i, wall, "†", ha="center", va="bottom", fontsize=11, color="#b00", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[r[0]] for r in rows], fontsize=9)
    ax.set_ylabel("Decode time (ms / token)", fontsize=11)
    ax.set_title(TITLE[plat], fontsize=12)
    ax.grid(axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, plat in zip(axes, ("orin", "thor")):
        panel(ax, plat)
    axes[0].legend(fontsize=8, loc="upper left", frameon=True)
    fig.text(0.5, 0.005, "† wall not decode-isolated (TRT-Edge via llm_bench; kernel/launch valid)",
             ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(a.out) / "fig04_nsys_breakdown"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
