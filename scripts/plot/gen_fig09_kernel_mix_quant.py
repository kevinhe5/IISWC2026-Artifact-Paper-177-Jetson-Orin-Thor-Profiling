#!/usr/bin/env python3
"""Fig 9 — Per-token GPU kernel-time decomposition across quantization formats
(4-bit / 8-bit / 16-bit) per framework, Llama-3.2-1B pp=128 gen=128 (AGX Orin | AGX Thor).
De-hardcoded: reads the shipped kernel_categories_quant JSONs; no embedded DATA.

The JSONs (data/nsys/kernel_categories_quant{,_thor}.json) hold, per (framework, quant),
the 5 paper kernel categories (ms / decode token) pre-bundled from the raw nsys per-kernel
attribution. Each stacked bar = matmul + attention + quantize + copy_cast + other.

  python3 gen_fig09_kernel_mix_quant.py [--out DIR]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
NSYS = REPO / "data" / "nsys"
SRC = {"orin": NSYS / "kernel_categories_quant.json",
       "thor": NSYS / "kernel_categories_quant_thor.json"}
TITLE = {"orin": "AGX Orin 32 GB", "thor": "AGX Thor 128 GB"}
FW_ORDER = ["trtllm", "llamacpp", "vllm", "sglang", "pytorch"]
FW_LABEL = {"trtllm": "TRT-LLM", "llamacpp": "llama.cpp", "vllm": "vLLM",
            "sglang": "SGLang", "pytorch": "PyTorch"}
QUANTS = ["fp16", "8bit", "4bit"]
QUANT_SHORT = {"fp16": "16b", "8bit": "8b", "4bit": "4b"}
CAT_ORDER = ["matmul", "attention", "quantize", "copy_cast", "other"]
CAT_LABEL = {"matmul": "matmul", "attention": "attention", "quantize": "quantize",
             "copy_cast": "copy/cast", "other": "other"}
CAT_COLOR = {"matmul": "#60a5fa", "attention": "#a78bfa", "quantize": "#f59e0b",
             "copy_cast": "#94a3b8", "other": "#cbd5e1"}


def panel(ax, plat):
    data = json.load(open(SRC[plat]))
    x = 0.0
    xticks, xlabels, group_marks = [], [], []
    for fw in FW_ORDER:
        d = data.get(fw)
        if not d:
            continue
        g0 = x
        for q in QUANTS:
            slot = d.get(q)
            if not slot:
                x += 1.0
                continue
            bottom = 0.0
            for c in CAT_ORDER:
                v = float(slot.get(c, 0.0))
                ax.bar(x, v, 0.85, bottom=bottom, color=CAT_COLOR[c],
                       label=CAT_LABEL[c] if (fw == FW_ORDER[0] and q == "fp16") else None)
                bottom += v
            ax.text(x, bottom, f"{bottom:.1f}", ha="center", va="bottom", fontsize=7)
            xticks.append(x); xlabels.append(QUANT_SHORT[q])
            x += 1.0
        group_marks.append(((g0 + x - 1.0) / 2.0, FW_LABEL[fw]))
        x += 0.6  # gap between framework groups
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=8)
    for cx, lbl in group_marks:
        ax.text(cx, -0.14, lbl, ha="center", va="top", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_ylabel("Kernel time (ms / token)", fontsize=11)
    ax.set_title(TITLE[plat], fontsize=12)
    ax.grid(axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, plat in zip(axes, ("orin", "thor")):
        panel(ax, plat)
    axes[0].legend(fontsize=8, loc="upper right", frameon=True, ncol=1)
    fig.subplots_adjust(bottom=0.16)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(a.out) / "fig09_kernel_mix_quant"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
