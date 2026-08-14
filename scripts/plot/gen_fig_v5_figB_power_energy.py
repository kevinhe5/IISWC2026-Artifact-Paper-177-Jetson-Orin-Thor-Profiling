#!/usr/bin/env python3
"""Generate fig07_power_energy.{pdf,png} — 2x2 panel:
   (a) Prefill power stack by Jetson hardware rail + MAXN TDP line
   (b) Prefill energy per prompt token (mJ / prompt-tok)
   (c) Decode  power stack by Jetson hardware rail + MAXN TDP line
   (d) Decode  energy per output token (mJ / output-tok)

Formatting preserved from the paper's original generator
(JetsonAnalysis/figs/scripts/gen_fig_v5_figB_power_energy.py).  Data
intake replaced: reads per-(framework, quant) rows from
data/chat/sweep_locked.csv filtered to Llama-3.2-1B, pp=4096, gen=4096,
AGX Orin locked clocks.

Power stacks use the three Jetson AGX Orin hardware power rails:
  * VDD_GPU_SOC : one physical rail powering GPU + iGPU + interconnect +
                  NVDLA + memory controllers. tegrastats reports the
                  modeled GPU and SOC components separately
                  (dec_gpu_mw / dec_soc_mw); we sum them here so the
                  stack reflects the actual hardware rail.
  * VDD_CPU_CV  : A78AE CPU cluster.
  * VDDQ_1V8_AO : LPDDR5 DRAM cells (bandwidth-derived estimate).
VIN_SYS_5V0 (system 5V / IO, <=5% of total on Orin) is not exposed by
tegrastats and is therefore omitted from the stacks.

Paper filename: fig_v5_figB_power_energy.pdf  (Orin-only, 1:1 candidate)

  python3 gen_fig_v5_figB_power_energy.py [--out DIR]
"""
import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC  = REPO / "data" / "chat" / "sweep_locked.csv"

TDP_W = 60.0   # AGX Orin 32 GB MAXN datasheet upper bound
PP    = 4096   # prompt tokens (for prefill energy normalization)
GEN   = 4096   # output tokens (for decode energy normalization)

# (fw_display_label, quant_display_label, csv_framework, csv_quantization)
# Order/labels match the paper's original figure; csv_quantization picks
# the row we want (e.g. vLLM uses its fp16_nocache row for apples-to-apples
# prefill BW; PyTorch 4bit corresponds to bnb-NF4 in the paper).
CELLS = [
    ("TRT-LLM",   "fp16",         "trtllm",   "fp16"),
    ("TRT-LLM",   "int4 (W4A16)", "trtllm",   "int4"),
    ("llama.cpp", "f16",          "llamacpp", "f16"),
    ("llama.cpp", "Q4_K_M",       "llamacpp", "Q4_K_M"),
    ("vLLM",      "fp16",         "vllm",     "fp16_nocache"),
    ("vLLM",      "gguf Q4_K_M",  "vllm",     "gguf_Q4_K_M"),
    ("SGLang",    "fp16",         "sglang",   "fp16"),
    ("PyTorch",   "bf16 (eager)", "pytorch",  "bf16"),
    ("PyTorch",   "bnb-NF4",      "pytorch",  "4bit"),
]

# Rail palette (matches the kernel-mix figure so the §IV suite shares one
# color language).
RAIL_COLOR = {
    "VDD_GPU_SOC": "#60a5fa",  # matmul blue - merged GPU + SoC fabric
    "VDD_CPU_CV":  "#f59e0b",  # quantize amber - Python-tax signature
    "VDDQ_LPDDR5": "#94a3b8",  # copy_cast slate - DRAM
}

# Framework palette for the energy bars — pale/pastel per paper original
# (color-picked from figs_original/fig_v5_figB_power_energy.pdf).
FW_COLOR_BAR = {
    "TRT-LLM":         "#AFC5B4",   # pale sage
    "llama.cpp":       "#E8B4B8",   # dusty rose
    "vLLM":            "#F7CB9F",   # pale apricot
    "SGLang":          "#C9BEDC",   # pale lavender
    "PyTorch":         "#D0B0A0",   # pale tan / beige
    "PyTorch+compile": "#B9C8AE",   # pale olive (near sage) — kept for continuity
}


def load_cells():
    """Return list of tuples matching the original DATA schema, sourced
    from data/chat/sweep_locked.csv."""
    # Build a lookup: (framework, quantization) -> row dict
    rows = {}
    for r in csv.DictReader(open(SRC)):
        if r["model"] != "Llama-3.2-1B": continue
        if int(r["prompt_tokens"]) != PP: continue
        if int(r["gen_tokens"]) != GEN: continue
        rows[(r["framework"], r["quantization"])] = r

    data = []
    for fw_lbl, q_lbl, csv_fw, csv_q in CELLS:
        r = rows.get((csv_fw, csv_q))
        if r is None:
            print(f"# WARN: missing sweep row for ({csv_fw}, {csv_q}); dropping")
            continue
        data.append((
            fw_lbl, q_lbl,
            float(r["pp_gpu_mw"]),  float(r["pp_cpu_mw"]),
            float(r["pp_soc_mw"]),  float(r["pp_dram_mw"]),
            float(r["pp_total_mw"]),
            float(r["prefill_energy_mj"]),
            float(r["dec_gpu_mw"]), float(r["dec_cpu_mw"]),
            float(r["dec_soc_mw"]), float(r["dec_dram_mw"]),
            float(r["dec_total_mw"]),
            float(r["decode_energy_mj"]),
        ))
    return data


def draw_power_stack(ax, x, gpu_soc, cpu, dram,
                     title, show_legend=False, ylim_top=None):
    """Stacked rail power (W) bars - 3 hardware rails + MAXN TDP line."""
    width = 0.66
    ax.bar(x, gpu_soc, width=width, color=RAIL_COLOR["VDD_GPU_SOC"],
           edgecolor="#0f1115", lw=0.6, zorder=3,
           label="GPU + SoC")
    ax.bar(x, cpu, width=width, bottom=gpu_soc,
           color=RAIL_COLOR["VDD_CPU_CV"], edgecolor="#0f1115", lw=0.6,
           zorder=3, label="CPU")
    ax.bar(x, dram, width=width, bottom=gpu_soc + cpu,
           color=RAIL_COLOR["VDDQ_LPDDR5"], edgecolor="#0f1115", lw=0.6,
           zorder=3, label="DRAM")

    # Paper original has no "N.N W" stack-top labels — dropped for parity.
    stack_top = gpu_soc + cpu + dram

    ax.axhline(TDP_W, color="#dc2626", lw=1.4, ls="--", zorder=5,
               label=f"MAXN TDP ({TDP_W:.0f} W)")
    ax.set_ylabel("Rail power (W)", fontsize=14, labelpad=8)
    ax.set_title(title, fontsize=14, pad=8, fontweight="bold")
    if ylim_top is None:
        ylim_top = max(TDP_W * 1.10, stack_top.max() * 1.22)
    ax.set_ylim(0, ylim_top)
    ax.grid(True, axis="y", color="#e5e7eb", lw=0.6, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    # Legend is drawn once at the figure level (horizontal band between rows).


def draw_energy_bar(ax, x, vals, fw_per_cell, title, ylabel):
    """Single energy per-token bar chart (color by framework)."""
    width = 0.66
    colors = [FW_COLOR_BAR.get(fw, "#888") for fw in fw_per_cell]
    ax.bar(x, vals, width=width, color=colors,
           edgecolor="#0f1115", lw=0.6, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.014, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, color="#1f2937",
                fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)
    ax.set_title(title, fontsize=14, pad=8, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.20)
    ax.grid(True, axis="y", color="#e5e7eb", lw=0.6, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="output directory")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    DATA = load_cells()
    if not DATA:
        raise SystemExit("no rows found in sweep_locked.csv for the requested cells")

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.6),
                             gridspec_kw={"hspace": 0.62, "wspace": 0.18})
    (axTL, axTR), (axBL, axBR) = axes

    x = np.arange(len(DATA))
    labels = [f"{d[0]}\n{d[1]}" for d in DATA]
    fws = [d[0] for d in DATA]

    # Prefill rail powers (W) - merge GPU + SOC into the VDD_GPU_SOC rail.
    pp_gpu_soc = np.array([d[2] + d[4] for d in DATA]) / 1000.0
    pp_cpu     = np.array([d[3]        for d in DATA]) / 1000.0
    pp_dram    = np.array([d[5]        for d in DATA]) / 1000.0
    pp_e_per_tok = np.array([d[7] / PP for d in DATA])

    # Decode rail powers (W) - same merge.
    dc_gpu_soc = np.array([d[8]  + d[10] for d in DATA]) / 1000.0
    dc_cpu     = np.array([d[9]          for d in DATA]) / 1000.0
    dc_dram    = np.array([d[11]         for d in DATA]) / 1000.0
    dc_e_per_tok = np.array([d[13] / GEN for d in DATA])

    pwr_top = max(TDP_W * 1.10,
                  (pp_gpu_soc + pp_cpu + pp_dram).max() * 1.22,
                  (dc_gpu_soc + dc_cpu + dc_dram).max() * 1.22)

    draw_power_stack(axTL, x, pp_gpu_soc, pp_cpu, pp_dram,
                     "(a) Prefill rail power",
                     show_legend=False, ylim_top=pwr_top)
    draw_energy_bar(axTR, x, pp_e_per_tok, fws,
                    "(b) Prefill energy / prompt token",
                    "mJ / prompt token")
    draw_power_stack(axBL, x, dc_gpu_soc, dc_cpu, dc_dram,
                     "(c) Decode rail power",
                     show_legend=False, ylim_top=pwr_top)
    draw_energy_bar(axBR, x, dc_e_per_tok, fws,
                    "(d) Decode energy / output token",
                    "mJ / output token")

    # Single horizontal legend band centered between the two rows,
    # matching the paper original.
    from matplotlib.patches import Patch
    from matplotlib.lines  import Line2D
    legend_handles = [
        Patch(facecolor=RAIL_COLOR["VDD_GPU_SOC"], edgecolor="#0f1115",
              lw=0.6, label="GPU + SoC"),
        Patch(facecolor=RAIL_COLOR["VDD_CPU_CV"], edgecolor="#0f1115",
              lw=0.6, label="CPU"),
        Patch(facecolor=RAIL_COLOR["VDDQ_LPDDR5"], edgecolor="#0f1115",
              lw=0.6, label="DRAM"),
        Line2D([0], [0], color="#dc2626", lw=1.6, ls="--",
               label=f"MAXN TDP ({TDP_W:.0f} W)"),
    ]
    fig.legend(handles=legend_handles, loc="center",
               bbox_to_anchor=(0.5, 0.505), ncol=4, frameon=False,
               fontsize=13, handlelength=1.8, handleheight=1.1,
               columnspacing=2.2, labelspacing=0.4)

    # x-tick labels on bottom row only (top row hidden)
    for ax in (axTL, axTR):
        ax.set_xticks(x)
        ax.set_xticklabels(["" for _ in labels])
    # Single-line x-tick labels (framework + quant on one slanted line).
    flat_labels = [f"{d[0]} {d[1]}" for d in DATA]
    for ax in (axBL, axBR):
        ax.set_xticks(x)
        ax.set_xticklabels(flat_labels, fontsize=11, rotation=28, ha="right")

    fig.tight_layout()
    pdf = out_dir / "fig07_power_energy.pdf"
    png = out_dir / "fig07_power_energy.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.14, dpi=200)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
