#!/usr/bin/env python3
"""fig8 (Thor pane) — kernel mix at short (gen=128) vs long (gen=65536) decode
context, Llama-3.2-1B, AGX Thor 128GB. Thor variant of Orin fig8_kernel_mix_longctx.py.

STATUS: long-context (gen=65536) decode traces are NOT yet captured on Thor
(PENDING GPU — the other agent captures gen=128 + gen=65536 with NVTX). This
script loads whatever kernel_categories_thor*.json files are present:
  - kernel_categories_thor.json          (short ctx, gen=128; produced this pass)
  - kernel_categories_thor_gen65536.json (long ctx; drop in when captured)
It plots the short-ctx bars now and the long-ctx bars once the second JSON
exists, using the same bundled 5-category stack as fig7. Frameworks without a
trace render as 'pending GPU'.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = os.path.join(HERE, "kernel_categories_thor.json")
LONG = os.path.join(HERE, "kernel_categories_thor_gen65536.json")
OUT_BASE = os.path.join(HERE, "fig_thor_fig8_kernel_mix_longctx")

FW_ORDER = ["trtedge_llm", "llamacpp", "vllm", "sglang", "pytorch"]
FW_LABEL = {"trtedge_llm": "TRT-Edge-LLM", "llamacpp": "llama.cpp", "vllm": "vLLM",
            "sglang": "SGLang", "pytorch": "PyTorch"}
BUNDLE5 = ["matmul", "attention", "quantize", "copy_cast", "other"]
COLORS = {"matmul": "#4C72B0", "attention": "#DD8452", "quantize": "#55A868",
          "copy_cast": "#C44E52", "other": "#8C8C8C"}


def bundle(cats):
    out = {k: 0.0 for k in BUNDLE5}
    for name, v in cats.items():
        ms = v["ms_per_tok"]
        out[name if name in out else "other"] += ms
    return out


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def main():
    short, long = load(SHORT), load(LONG)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(FW_ORDER)), 4))
    width = 0.38
    for xi, fw in enumerate(FW_ORDER):
        for off, (src, tag) in enumerate([(short, "short"), (long, "long")]):
            xpos = xi + (off - 0.5) * width
            if fw in src and isinstance(src[fw], dict) and "categories" in src[fw]:
                b = bundle(src[fw]["categories"]); bottom = 0.0
                for cat in BUNDLE5:
                    ax.bar(xpos, b[cat], width=width, bottom=bottom, color=COLORS[cat])
                    bottom += b[cat]
            else:
                ax.text(xpos, 0.5, "pending\nGPU", ha="center", va="center",
                        fontsize=6, color="#999")
    ax.set_xticks(np.arange(len(FW_ORDER)))
    ax.set_xticklabels([FW_LABEL[fw] for fw in FW_ORDER], rotation=15)
    ax.set_ylabel("GPU kernel ms / decode-tok")
    ax.set_title("Fig 8 (Thor pane) — short (left) vs long-ctx (right); long-ctx PENDING GPU")
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".png", dpi=150)
    fig.savefig(OUT_BASE + ".pdf")
    print("wrote", OUT_BASE + ".{png,pdf}", "| long-ctx present:", bool(long))


if __name__ == "__main__":
    main()
