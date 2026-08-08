#!/usr/bin/env python3
"""fig7 (Thor pane) — per-framework GPU kernel-time decomposition, Llama-3.2-1B,
AGX Thor 128GB (sm_110). Thor variant of Orin fig7_kernel_mix_quant.py.

Unlike the Orin script (which embeds values inline), this loads the extracted
Thor categories from kernel_categories_thor.json so the figure tracks the real
trace extraction. Each framework bar is the bundled 5-category stack
(matmul / attention / quantize / copy_cast / other); 'other' folds
norm+rope+kvcache+activation+elementwise+sampling.

Availability on Thor (see the JSON's _pending_gpu): only vLLM, llama.cpp and
TRT-Edge-LLM (int4_awq engine) decode traces exist. SGLang and PyTorch decode
traces are PENDING GPU (other agent). Per-quantization sub-bars need the fp16
+ 4-bit trace pair per engine; only the default variant is traced here, so this
renders one bar per available framework. Extend once the pending traces land.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "kernel_categories_thor.json")
OUT_BASE = os.path.join(HERE, "fig_thor_fig7_kernel_mix_quant")

FW_ORDER = ["trtedge_llm", "llamacpp", "vllm", "sglang", "pytorch"]
FW_LABEL = {"trtedge_llm": "TRT-Edge-LLM", "llamacpp": "llama.cpp", "vllm": "vLLM",
            "sglang": "SGLang", "pytorch": "PyTorch"}
BUNDLE5 = ["matmul", "attention", "quantize", "copy_cast", "other"]
FOLD_OTHER = {"norm", "rope", "kvcache", "activation", "elementwise", "sampling", "other"}
COLORS = {"matmul": "#4C72B0", "attention": "#DD8452", "quantize": "#55A868",
          "copy_cast": "#C44E52", "other": "#8C8C8C"}


def bundle(cats):
    out = {k: 0.0 for k in BUNDLE5}
    for name, v in cats.items():
        ms = v["ms_per_tok"]
        if name in ("matmul", "attention", "quantize", "copy_cast"):
            out[name] += ms
        else:  # everything in FOLD_OTHER
            out["other"] += ms
    return out


def main():
    data = json.load(open(JSON))
    avail = [fw for fw in FW_ORDER if fw in data and isinstance(data[fw], dict)
             and "categories" in data[fw]]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(FW_ORDER)), 4))
    x = np.arange(len(FW_ORDER))
    for xi, fw in enumerate(FW_ORDER):
        if fw not in avail:
            ax.text(xi, 0.5, "pending\nGPU", ha="center", va="center", fontsize=8, color="#999")
            continue
        b = bundle(data[fw]["categories"])
        bottom = 0.0
        for cat in BUNDLE5:
            ax.bar(xi, b[cat], bottom=bottom, color=COLORS[cat], width=0.6,
                   label=cat if xi == avail.index(avail[0]) else None)
            bottom += b[cat]
        ax.text(xi, bottom + 0.1, f"{bottom:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([FW_LABEL[fw] for fw in FW_ORDER], rotation=15)
    ax.set_ylabel("GPU kernel ms / decode-tok")
    ax.set_title("Fig 7 (Thor pane) — kernel mix, Llama-3.2-1B, AGX Thor 128GB")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".png", dpi=150)
    fig.savefig(OUT_BASE + ".pdf")
    print("wrote", OUT_BASE + ".{png,pdf}", "| available fw:", avail,
          "| pending:", data.get("_pending_gpu"))


if __name__ == "__main__":
    main()
