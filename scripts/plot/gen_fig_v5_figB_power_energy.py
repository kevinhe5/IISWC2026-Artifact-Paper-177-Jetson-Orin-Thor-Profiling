#!/usr/bin/env python3
"""Generate fig_v5_figB_power_energy.{pdf,png}
   - 2x2 panel:
       (a) Prefill power stack by Jetson hardware rail + MAXN TDP line
       (b) Prefill energy per prompt token (mJ / prompt-tok)
       (c) Decode  power stack by Jetson hardware rail + MAXN TDP line
       (d) Decode  energy per output token (mJ / output-tok)

Power stacks use the three Jetson AGX Orin hardware power rails (matching
gen_fig_v5_fig9_energy_rails.py):
  * VDD_GPU_SOC : one physical rail powering GPU + iGPU + interconnect +
                  NVDLA + memory controllers. tegrastats reports the
                  modeled GPU and SOC components separately
                  (dec_gpu_mw / dec_soc_mw); we sum them here so the
                  stack reflects the actual hardware rail.
  * VDD_CPU_CV  : A78AE CPU cluster.
  * VDDQ_1V8_AO : LPDDR5 DRAM cells (bandwidth-derived estimate).
VIN_SYS_5V0 (system 5V / IO, <=5% of total on Orin) is not exposed by
our tegrastats build and is therefore omitted from the stacks.

Model:  Llama-3.2-1B-Instruct
Cond:   pp=4096, gen=4096, AGX Orin 32 GB, locked clocks
        - sustained-workload cell where both prefill and decode reach
          steady-state on the Tegra power rails. At short prefill
          (pp<=256, TTFT<50 ms) tegrastats' 25 ms sampling undersamples
          the GPU voltage/frequency ramp, biasing P_pref low by 10-40%;
          pp=4096 (TTFT 200-1600 ms across fw) avoids that artifact.
        - vLLM/SGLang use their `fp16_nocache` rows (no prefix caching)
          so the prefill phase actually runs the matmul instead of
          short-circuiting through the radix cache.

Source CSV (kept for traceability):
  data/sweep_results/sweep_locked_20260428_020532.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
OUT_PDF = _HERE / "fig_v5_figB_power_energy.pdf"
OUT_PNG = _HERE / "fig_v5_figB_power_energy.png"

TDP_W = 60.0   # AGX Orin 32 GB MAXN datasheet upper bound

PP = 4096      # prompt tokens (for prefill energy normalization)
GEN = 4096     # output tokens (for decode energy normalization)

# De-hardcoded: DATA is loaded from artifact/data/chat/sweep_locked.csv
# (Orin main-sweep CSV). One row per (framework, quantization) cell at
# pp=4096, gen=4096. Falls back to embedded values only if the CSV path
# cannot be found (for the paper's original submission-time values).
#
# Cells extracted (framework, quant_in_csv, display_label):
CELLS = [
    ("trtllm",   "fp16",         "TRT-LLM",   "fp16"),
    ("trtllm",   "int4",         "TRT-LLM",   "int4 (W4A16)"),
    ("llamacpp", "f16",          "llama.cpp", "f16"),
    ("llamacpp", "Q4_K_M",       "llama.cpp", "Q4_K_M"),
    ("vllm",     "fp16_nocache", "vLLM",      "fp16"),  # no-cache variant
    ("vllm",     "gguf_Q4_K_M",  "vLLM",      "gguf Q4_K_M"),
    ("sglang",   "fp16_nocache", "SGLang",    "fp16"),  # no-cache variant
    ("pytorch",  "bf16",         "PyTorch",   "bf16 (eager)"),
    ("pytorch",  "4bit",         "PyTorch",   "bnb-NF4"),
]


# Fallback for cells not present in the CSV (e.g. SGLang on Orin isn't in
# sweep_locked.csv; it lives in sweep_locked_orin_15run.csv once rep15
# finishes). These are the paper's original submission-time values.
_HARDCODED_FALLBACK = {
    ("SGLang", "fp16"):
        (19400, 3444, 7665, 2784, 33293,   7531.34,
         21353, 3318,11394, 5441, 41506,4006099.57),
}


def _load_data_from_csv(csv_path, pp=PP, gen=GEN):
    """Read sweep_locked{,_orin_15run}.csv, extract the CELLS at (pp, gen),
    assemble DATA tuples with the same schema as the pre-de-hardcode
    version. For cells absent from the CSV (e.g. SGLang on Orin at
    submission time), fall back to `_HARDCODED_FALLBACK` and print a
    stderr note listing every fallback."""
    import csv, sys
    rows_by_key = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                r_pp = int(r["prompt_tokens"])
                r_gen = int(r["gen_tokens"])
            except (KeyError, ValueError):
                continue
            if r_pp != pp or r_gen != gen:
                continue
            key = (r["framework"], r["quantization"])
            if key in rows_by_key:
                continue  # first match wins
            rows_by_key[key] = r

    def _num(row, col, default=0.0):
        v = row.get(col, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    out, missing = [], []
    for fw_csv, quant_csv, fw_disp, quant_disp in CELLS:
        r = rows_by_key.get((fw_csv, quant_csv))
        if r is None:
            fb = _HARDCODED_FALLBACK.get((fw_disp, quant_disp))
            if fb is None:
                raise KeyError(f"missing cell in {csv_path}: fw={fw_csv} "
                               f"quant={quant_csv} pp={pp} gen={gen} "
                               "and no hardcoded fallback")
            missing.append((fw_disp, quant_disp))
            out.append((fw_disp, quant_disp, *fb))
        else:
            out.append((
                fw_disp, quant_disp,
                _num(r, "pp_gpu_mw"),  _num(r, "pp_cpu_mw"),
                _num(r, "pp_soc_mw"),  _num(r, "pp_dram_mw"),
                _num(r, "pp_total_mw"),
                _num(r, "prefill_energy_mj"),
                _num(r, "dec_gpu_mw"), _num(r, "dec_cpu_mw"),
                _num(r, "dec_soc_mw"), _num(r, "dec_dram_mw"),
                _num(r, "dec_total_mw"),
                _num(r, "decode_energy_mj"),
            ))
    if missing:
        print(f"[gen_fig_v5_figB_power_energy] using hardcoded fallback for "
              f"{len(missing)} cell(s) absent from {csv_path.name}: "
              f"{missing}", file=sys.stderr)
    return out


# Locate the shipped Orin sweep CSV — prefer 15-run aggregate if present.
_ART = _HERE.parent.parent   # scripts/plot/ → scripts/ → artifact/
for _cand in (_ART / "data/chat/sweep_locked.csv",):
    if _cand.is_file():
        _CSV_PATH = _cand
        break
else:
    raise FileNotFoundError(
        f"sweep_locked{{,_orin_15run}}.csv not found under {_ART}/data/chat/. "
        "This generator requires the Orin main-sweep CSV.")

DATA = _load_data_from_csv(_CSV_PATH)

# Rail palette (matches gen_fig_v5_fig9_energy_rails.py, derived from the
# kernel-mix figure so the §IV figure suite shares one color language).
RAIL_COLOR = {
    "VDD_GPU_SOC": "#60a5fa",  # matmul blue  - merged GPU + SoC fabric
    "VDD_CPU_CV":  "#f59e0b",  # quantize amber - Python-tax signature
    "VDDQ_LPDDR5": "#94a3b8",  # copy_cast slate - DRAM
}

# Framework palette for the energy bars (matches the Pareto figure).
FW_COLOR_BAR = {
    "TRT-LLM":         "#60a5fa",
    "llama.cpp":       "#f59e0b",
    "vLLM":            "#a78bfa",
    "SGLang":          "#94a3b8",
    "PyTorch":         "#ef4444",
    "PyTorch+compile": "#16a34a",
}


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

    stack_top = gpu_soc + cpu + dram
    for xi, st in zip(x, stack_top):
        ax.text(xi, st + 1.2, f"{st:.1f} W",
                ha="center", va="bottom", fontsize=10,
                color="#1f2937", fontweight="bold")

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
    if show_legend:
        ax.legend(loc="upper right", fontsize=11, frameon=True,
                  framealpha=0.95, handlelength=1.4, handleheight=1.1,
                  labelspacing=0.4, ncol=1)


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
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.2),
                             gridspec_kw={"hspace": 0.46, "wspace": 0.18})
    (axTL, axTR), (axBL, axBR) = axes

    x = np.arange(len(DATA))
    labels = [f"{d[0]}\n{d[1]}" for d in DATA]
    fws = [d[0] for d in DATA]

    # Prefill rail powers (W) - merge GPU + SOC into the VDD_GPU_SOC rail.
    pp_gpu_soc = np.array([d[2] + d[4] for d in DATA]) / 1000.0
    pp_cpu = np.array([d[3] for d in DATA]) / 1000.0
    pp_dram = np.array([d[5] for d in DATA]) / 1000.0
    pp_e_per_tok = np.array([d[7] / PP for d in DATA])

    # Decode rail powers (W) - same merge.
    dc_gpu_soc = np.array([d[8] + d[10] for d in DATA]) / 1000.0
    dc_cpu = np.array([d[9] for d in DATA]) / 1000.0
    dc_dram = np.array([d[11] for d in DATA]) / 1000.0
    dc_e_per_tok = np.array([d[13] / GEN for d in DATA])

    pwr_top = max(TDP_W * 1.10,
                  (pp_gpu_soc + pp_cpu + pp_dram).max() * 1.22,
                  (dc_gpu_soc + dc_cpu + dc_dram).max() * 1.22)

    draw_power_stack(axTL, x, pp_gpu_soc, pp_cpu, pp_dram,
                     "(a) Prefill rail power",
                     show_legend=True, ylim_top=pwr_top)
    draw_energy_bar(axTR, x, pp_e_per_tok, fws,
                    "(b) Prefill energy / prompt token",
                    "mJ / prompt token")
    draw_power_stack(axBL, x, dc_gpu_soc, dc_cpu, dc_dram,
                     "(c) Decode rail power",
                     show_legend=False, ylim_top=pwr_top)
    draw_energy_bar(axBR, x, dc_e_per_tok, fws,
                    "(d) Decode energy / output token",
                    "mJ / output token")

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
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.14, dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
