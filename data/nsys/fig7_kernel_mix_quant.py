#!/usr/bin/env python3
"""Generate fig_v5_fig7_kernel_mix_quant.{pdf,png}
   — Per-framework × per-quantization GPU kernel-time decomposition.
     Answers: "How does the kernel mix shift when we drop quantization?"

Layout: 5 framework groups, each containing up to 3 vertical stacked bars
(Q4 / Q8 / fp16). Each bar is stacked into 5 kernel categories:
  matmul · attention · quantize · copy_cast · other
The 'other' bucket lumps norm + rope + kvcache + activation + elementwise + sampling.

Mirrors the visual style of fig_v5_fig6_nsys_breakdown.pdf
(group labels, unprof-tick is replaced by total-ms label at top).

Source CSV/JSON (kept for traceability — values below are extracted
from the dashboard's per-kernel attribution at pp=128, gen=128,
Llama-3.2-1B, AGX Orin 32 GB locked clocks):
  data/nsys_profiles/kernel_categories_compare.json

Self-contained: all values embedded below.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_BASE  = "/nvme/ispass/paper_jetson/JetsonAnalysis/figs/fig_v5_fig7_kernel_mix_quant"

# Frameworks and quant slots, in display order
FW_ORDER  = ["trtllm", "llamacpp", "vllm", "sglang", "pytorch"]
FW_LABEL  = {"trtllm":"TRT-LLM", "llamacpp":"llama.cpp", "vllm":"vLLM",
             "sglang":"SGLang",  "pytorch":"PyTorch"}
QUANTS    = ["4bit", "8bit", "fp16"]
QUANT_LBL = {"4bit":"4-bit", "8bit":"8-bit", "fp16":"16-bit"}

# Per-framework short quant labels (concise, paper-friendly).
# Overrides the dashboard's "gguf_Q4_K_M" / "fp16 (HF)" verbose strings.
# Kept ≤ 7 chars so the two-line x-tick stays inside its bar width.
QUANT_SHORT = {
    "trtllm":   {"4bit": "W4A16",  "8bit": "W8 SmQ",  "fp16": "fp16"},
    "llamacpp": {"4bit": "Q4_K_M", "8bit": "Q8_0",    "fp16": "F16"},
    "vllm":     {"4bit": "Q4 GGUF","8bit": "Q8 GGUF", "fp16": "fp16 HF"},
    "sglang":   {"4bit": "Q4 GGUF","8bit": "Q8 GGUF", "fp16": "fp16 HF"},
    "pytorch":  {"4bit": "NF4",    "8bit": "bnb-i8",  "fp16": "bf16"},
}

# Category bundling: dashboard's 10 → paper's 5
CAT_BUNDLE = {
    "matmul":     ["matmul"],
    "attention":  ["attention"],
    "quantize":   ["quantize"],
    "copy_cast":  ["copy_cast"],
    "other":      ["norm", "rope", "kvcache", "activation", "elementwise", "sampling"],
}
CAT_ORDER  = ["matmul", "attention", "quantize", "copy_cast", "other"]
CAT_LABEL  = {
    "matmul":    "matmul  (weight-stream)",
    "attention": "attention  (KV-stream)",
    "quantize":  "quantize / dequant",
    "copy_cast": "copy / cast",
    "other":     "other  (norm + rope + kv-write + ...)",
}
CAT_COLOR  = {
    "matmul":    "#60a5fa",
    "attention": "#a78bfa",
    "quantize":  "#f59e0b",
    "copy_cast": "#94a3b8",
    "other":     "#cbd5e1",
}


# =====================================================================
# Hardcoded data — pre-bundled into the 5 paper categories from the raw
# 10-category dashboard JSON.  Each (fw, quant) cell is ms / decode token,
# Llama-3.2-1B at pp=128, gen=128, AGX Orin 32 GB locked clocks.
# Snapshot date: 2026-05-15 (kernel_categories_compare.json).
# =====================================================================
DATA = {
    "trtllm": {
        "labels": {"4bit": "int4 (W4A16)", "8bit": "int8 (SmoothQuant)", "fp16": "fp16"},
        "4bit":   {"matmul":  6.182, "attention": 0.253, "quantize": 0.000, "copy_cast": 0.010, "other": 0.557},
        "8bit":   {"matmul":  8.944, "attention": 0.250, "quantize": 0.000, "copy_cast": 0.010, "other": 0.593},
        "fp16":   {"matmul": 14.228, "attention": 0.253, "quantize": 0.000, "copy_cast": 0.000, "other": 0.610},
    },
    "llamacpp": {
        "labels": {"4bit": "Q4_K_M", "8bit": "Q8_0", "fp16": "f16"},
        "4bit":   {"matmul":  8.358, "attention": 1.003, "quantize": 0.198, "copy_cast": 0.349, "other": 0.423},
        "8bit":   {"matmul": 12.442, "attention": 1.009, "quantize": 0.204, "copy_cast": 0.349, "other": 0.428},
        "fp16":   {"matmul": 17.962, "attention": 1.056, "quantize": 0.000, "copy_cast": 0.345, "other": 0.426},
    },
    "vllm": {
        "labels": {"4bit": "gguf_Q4_K_M", "8bit": "gguf_Q8_0", "fp16": "fp16 (HF)"},
        "4bit":   {"matmul":  9.642, "attention": 0.268, "quantize": 0.296, "copy_cast": 0.173, "other": 0.429},
        "8bit":   {"matmul": 10.402, "attention": 0.269, "quantize": 0.333, "copy_cast": 0.177, "other": 0.446},
        "fp16":   {"matmul": 14.679, "attention": 0.193, "quantize": 0.000, "copy_cast": 0.015, "other": 0.569},
    },
    "sglang": {
        "labels": {"4bit": "gguf_Q4_K_M", "8bit": "gguf_Q8_0", "fp16": "fp16 (HF)"},
        "4bit":   {"matmul":  9.813, "attention": 0.220, "quantize": 0.302, "copy_cast": 0.152, "other": 0.511},
        "8bit":   {"matmul": 10.650, "attention": 0.222, "quantize": 0.329, "copy_cast": 0.153, "other": 0.535},
        "fp16":   {"matmul": 15.464, "attention": 0.222, "quantize": 0.000, "copy_cast": 0.166, "other": 0.441},
    },
    "pytorch": {
        "labels": {"4bit": "bnb-NF4", "8bit": "bnb-int8", "fp16": "bf16"},
        "4bit":   {"matmul":  9.513, "attention": 0.242, "quantize": 0.000, "copy_cast": 0.439, "other": 1.404},
        "8bit":   {"matmul": 15.580, "attention": 0.233, "quantize": 0.843, "copy_cast": 0.619, "other": 2.542},
        "fp16":   {"matmul": 15.150, "attention": 0.241, "quantize": 0.000, "copy_cast": 0.445, "other": 1.538},
    },
}


def _load():
    """Reshape hardcoded DATA into the row-list the renderer expects."""
    rows = []
    for fw in FW_ORDER:
        if fw not in DATA: continue
        labs = DATA[fw].get("labels", {})
        for q in QUANTS:
            slot = DATA[fw].get(q)
            if not slot:
                rows.append({"fw":fw, "q":q, "label":None, "total":0.0,
                             "cats":{c:0.0 for c in CAT_ORDER}, "missing":True})
                continue
            cats = {c: slot.get(c, 0.0) for c in CAT_ORDER}
            rows.append({"fw":fw, "q":q, "label":labs.get(q, q),
                         "total":sum(cats.values()),
                         "cats":cats, "missing":False})
    return rows


def main():
    rows = _load()

    # X positions: 5 fw groups × 3 quant bars per group, gap between groups
    width      = 0.62
    group_gap  = 1.6   # wider gap so x-tick labels don't collide across groups
    n_per_fw   = len(QUANTS)
    xs, fw_mid = [], []
    cursor = 0.0
    for fw in FW_ORDER:
        x_fw = [cursor + i for i in range(n_per_fw)]
        xs.extend(x_fw)
        fw_mid.append((x_fw[0] + x_fw[-1]) / 2)
        cursor = x_fw[-1] + group_gap + 1
    xs = np.array(xs)

    fig, ax = plt.subplots(figsize=(12.0, 5.6))

    # Stacked bars
    y_max = max(r["total"] for r in rows) * 1.20
    legend_handles = {}
    for i, r in enumerate(rows):
        if r["missing"]:
            ax.text(xs[i], 0.5, "n/a", ha="center", va="bottom",
                    fontsize=9, color="#9ca3af", fontstyle="italic")
            continue
        bottom = 0.0
        for cat in CAT_ORDER:
            v = r["cats"][cat]
            if v <= 0: continue
            h = ax.bar(xs[i], v, width=width, bottom=bottom,
                       color=CAT_COLOR[cat], edgecolor="#0f1115",
                       linewidth=0.4, zorder=3,
                       label=CAT_LABEL[cat] if cat not in legend_handles else None)
            if cat not in legend_handles: legend_handles[cat] = h
            # Inner label only if segment is tall enough
            if v >= y_max * 0.04:
                ax.text(xs[i], bottom + v/2, f"{v:.1f}",
                        ha="center", va="center", fontsize=8.0,
                        color="#1f2937", fontweight="bold")
            bottom += v
        # Total label at top of bar
        ax.text(xs[i], bottom + y_max*0.012, f"{r['total']:.1f}",
                ha="center", va="bottom", fontsize=9.5,
                color="#1f2937", fontweight="bold")

    # X ticks: short quant label (line 1: bit-width, line 2: framework's
    # concrete quant scheme).
    quant_labels = []
    for r in rows:
        short = QUANT_SHORT.get(r["fw"], {}).get(r["q"], r["q"])
        quant_labels.append(f"{QUANT_LBL[r['q']]}\n{short}")
    ax.set_xticks(xs)
    ax.set_xticklabels(quant_labels, fontsize=9, linespacing=1.15)

    # Group labels (framework name above each triple)
    for fw, mid in zip(FW_ORDER, fw_mid):
        ax.text(mid, y_max * 1.05, FW_LABEL[fw], ha="center",
                va="bottom", fontsize=15, fontweight="bold", color="#1f2937")
    # Light separators
    for i in range(len(FW_ORDER) - 1):
        sep = (fw_mid[i] + fw_mid[i+1]) / 2
        ax.axvline(sep, color="#d4d4d8", linewidth=0.8, linestyle="--",
                   zorder=1, alpha=0.7)

    ax.set_ylim(0, y_max * 1.05)
    ax.set_ylabel("GPU kernel time (ms / decode token)", fontsize=14, labelpad=10)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.5, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)

    # Legend
    ax.legend(loc="upper right", fontsize=10, frameon=False,
              handlelength=1.4, handleheight=1.1, labelspacing=0.4)

    fig.tight_layout(pad=0.7)
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", pad_inches=0.18, dpi=200)
    print(f"wrote {OUT_BASE}.pdf")
    print(f"wrote {OUT_BASE}.png")


if __name__ == "__main__":
    main()
