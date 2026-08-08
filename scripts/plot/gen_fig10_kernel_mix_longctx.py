#!/usr/bin/env python3
"""Fig 10 — Per-token GPU kernel-time decomposition at short (gen=128) vs long
(gen=65 536) decoded context, per framework, Llama-3.2-1B (AGX Orin above, AGX Thor below).
De-hardcoded: reads the shipped kernel_categories_longctx JSONs; no embedded DATA.

Each panel: 5 frameworks × 4 bars (Q4-short, Q4-long, fp16-short, fp16-long), each bar
stacked into 6 categories. Missing cells (OOM / not captured) are left blank. The long-context
growth is dominated by the attention category (KV grows with context).

  python3 gen_fig10_kernel_mix_longctx.py [--out DIR]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
NSYS = REPO / "data" / "nsys"
SRC = {"orin": NSYS / "kernel_categories_longctx.json",
       "thor": NSYS / "kernel_categories_longctx_thor.json"}
TITLE = {"orin": "AGX Orin 32 GB", "thor": "AGX Thor 128 GB"}
FW_ORDER = ["trtllm", "llamacpp", "vllm", "sglang", "pytorch"]
FW_LABEL = {"trtllm": "TRT-LLM", "llamacpp": "llama.cpp", "vllm": "vLLM",
            "sglang": "SGLang", "pytorch": "PyTorch"}
QUANT_ORDER = ["Q4", "fp16"]
CTX_ORDER = ["short_ctx", "long_ctx"]
CTX_SHORT = {"short_ctx": "s", "long_ctx": "L"}
CAT_ORDER = ["matmul", "attention", "quantize", "copy_cast", "elementwise", "other"]
CAT_LABEL = {"matmul": "matmul", "attention": "attention", "quantize": "quantize",
             "copy_cast": "copy/cast", "elementwise": "elementwise", "other": "other"}
CAT_COLOR = {"matmul": "#60a5fa", "attention": "#a78bfa", "quantize": "#f59e0b",
             "copy_cast": "#94a3b8", "elementwise": "#34d399", "other": "#cbd5e1"}


def _cats(slot):
    """Return the 6-category vector for a ctx slot, or None if missing/empty."""
    if not isinstance(slot, dict):
        return None
    vals = [slot.get(c) for c in CAT_ORDER]
    if all(v is None for v in vals) or sum(v or 0.0 for v in vals) == 0.0:
        return None
    return [float(v or 0.0) for v in vals]


def panel(ax, plat):
    data = json.load(open(SRC[plat]))
    x = 0.0
    xticks, xlabels, group_marks = [], [], []
    for fw in FW_ORDER:
        d = data.get(fw)
        if not d:
            continue
        g0 = x
        for q in QUANT_ORDER:
            for ctx in CTX_ORDER:
                slot = (d.get(q) or {}).get(ctx)
                cats = _cats(slot)
                if cats is None:
                    x += 1.0
                    continue
                bottom = 0.0
                for c, v in zip(CAT_ORDER, cats):
                    ax.bar(x, v, 0.85, bottom=bottom, color=CAT_COLOR[c],
                           label=CAT_LABEL[c] if (fw == FW_ORDER[0] and q == "Q4" and ctx == "short_ctx") else None)
                    bottom += v
                ax.text(x, bottom, f"{bottom:.0f}", ha="center", va="bottom", fontsize=6.5)
                xticks.append(x); xlabels.append(f"{q}\n{CTX_SHORT[ctx]}")
                x += 1.0
        group_marks.append(((g0 + x - 1.0) / 2.0, FW_LABEL[fw]))
        x += 0.6
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=6.5)
    for cx, lbl in group_marks:
        ax.text(cx, -0.22, lbl, ha="center", va="top", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_ylabel("Kernel time (ms / token)", fontsize=10)
    ax.set_title(TITLE[plat], fontsize=12)
    ax.grid(axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0))
    for ax, plat in zip(axes, ("orin", "thor")):
        panel(ax, plat)
    axes[0].legend(fontsize=7.5, loc="upper left", frameon=True, ncol=2)
    fig.subplots_adjust(hspace=0.45, bottom=0.09)
    out = Path(a.out) / "fig10_kernel_mix_longctx"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
