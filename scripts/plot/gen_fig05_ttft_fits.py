#!/usr/bin/env python3
"""Fig 5 — Prefill TTFT vs input length, Llama-3.2-1B fp16.
LEFT panel: AGX Thor 128 GB (LPDDR5X 273 GB/s), 7 series.
RIGHT panel: AGX Orin 32 GB (LPDDR5 204.8 GB/s), 6 series (no FA-on split).

Formatting reproduced from figs_original/fig_v5_ttft_fits_new_sweep.pdf:
two panels (Thor first), per-framework scatter + SOLID quadratic fit
curves, markers with dark edges, monospace legend with a right-aligned
"Thor Fit MAPE" / "Orin Fit MAPE" column, formula box in the upper-right
of the Orin panel, linear-scale y, x ticks at 128/256/512/1K/2K/4K,
shared "Prefill TTFT (ms)" ylabel.

Data intake reads:
    data/chat/sweep_locked_thor.csv     (Thor: 7 fw incl. llamacpp_fa +
                                         pytorch_compile)
    data/chat/sweep_locked.csv          (Orin: 5 fw baseline)
    data/chat/pytorch_compile.csv       (Orin: adds pytorch_compile row)
All filtered to Llama-3.2-1B, fp16-family quants, per-cell averaged.

Paper filename: fig_v5_ttft_fits_new_sweep.pdf

  python3 gen_fig05_ttft_fits.py [--out DIR]
"""
import argparse, csv, math
from collections import defaultdict, OrderedDict
from pathlib import Path
from statistics import fmean, mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SWEEP_ORIN = REPO / "data" / "chat" / "sweep_locked.csv"
SWEEP_THOR = REPO / "data" / "chat" / "sweep_locked_thor.csv"
COMPILE_ORIN = REPO / "data" / "chat" / "pytorch_compile.csv"

FP16 = {"fp16", "bf16", "f16", "16-bit"}

# Ordered per-panel: trt on top, PyTorch(compile) last, matching paper legend.
ORDER_THOR = ["trtedge_llm", "vllm", "sglang", "llamacpp_fa",
              "llamacpp", "pytorch", "pytorch_compile"]
ORDER_ORIN = ["trtllm", "vllm", "sglang", "llamacpp_fa",
              "llamacpp", "pytorch", "pytorch_compile"]

# Human-readable names (paper wording).
LABEL = {
    "trtllm":          "TensorRT-LLM",
    "trtedge_llm":     "TensorRT-Edge-LLM",
    "vllm":            "vLLM",
    "sglang":          "SGLang",
    "llamacpp_fa":     "llama.cpp (FA on)",
    "llamacpp":        "llama.cpp",
    "pytorch":         "PyTorch (eager)",
    "pytorch_compile": "PyTorch(compile)",
}

# Palette color-picked from paper PDF (each fw has its own colour).
COLOR = {
    "trtllm":          "#1f77b4", "trtedge_llm":     "#1f77b4",
    "vllm":            "#ff7f0e",
    "sglang":          "#9467bd",
    "llamacpp_fa":     "#22c1d1",   # cyan
    "llamacpp":        "#e377c2",   # magenta
    "pytorch":         "#2ca02c",
    "pytorch_compile": "#a56a13",   # brown
}
MARKER = {
    "trtllm": "o", "trtedge_llm": "o",
    "vllm": "s", "sglang": "^",
    "llamacpp_fa": "D", "llamacpp": "D",
    "pytorch": "v", "pytorch_compile": "X",
}


def _fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def _i_pad(i):
    return int(math.ceil(i / 128.0) * 128)


def load_rows(csv_path, model="Llama-3.2-1B"):
    """{fw: [{I, ttft}, ...]} averaged per I across cells matching the
    fp16 family of quants."""
    by = defaultdict(list)
    for r in csv.DictReader(open(csv_path, newline="")):
        if (r.get("quantization") or "").strip() not in FP16: continue
        if (r.get("model") or "").strip() != model: continue
        fw = r.get("framework")
        I = _fnum(r.get("prompt_tokens"))
        ttft = _fnum(r.get("ttft_ms"))
        if I is None or ttft is None: continue
        by[fw].append((int(I), ttft))
    return by


def fit_prefill(pts):
    """Original quadratic fit: L(I_p) = a I_p^2 + b I_p + c. Averages TTFT
    within each I."""
    by_I = defaultdict(list)
    for I, ttft in pts:
        by_I[I].append(ttft)
    Is = sorted(by_I)
    if len(Is) < 3: return None
    Ip = np.array([_i_pad(i) for i in Is], dtype=float)
    Tm = np.array([mean(by_I[i]) for i in Is], dtype=float)
    coef, *_ = np.linalg.lstsq(
        np.column_stack([Ip * Ip, Ip, np.ones_like(Ip)]), Tm, rcond=None)
    a, b, c = coef
    pred = a * Ip * Ip + b * Ip + c
    mape = 100.0 * fmean(abs(p - m) / m for m, p in zip(Tm, pred) if m)
    return {"a": a, "b": b, "c": c, "Is": Is, "Ip": Ip, "Tm": Tm,
            "pred": pred, "mape": mape}


def draw_panel(ax, rows_by_fw, order, title, legend_title, show_ylabel,
               show_formula_box):
    fits = {}
    for fw in order:
        pts = rows_by_fw.get(fw)
        if not pts: continue
        f = fit_prefill(pts)
        if f is None: continue
        fits[fw] = f
        ax.scatter(f["Ip"], f["Tm"], color=COLOR[fw], marker=MARKER[fw],
                   s=48, zorder=3, edgecolor="#1f2937", linewidths=0.6)
        I_grid = np.linspace(min(f["Ip"]), max(f["Ip"]), 200)
        T_fit = f["a"] * I_grid * I_grid + f["b"] * I_grid + f["c"]
        ax.plot(I_grid, T_fit, color=COLOR[fw], linestyle="-",
                linewidth=1.5, alpha=0.95, zorder=2)

    ax.set_xlabel(r"Input sequence length $I$ (tokens)", fontsize=12)
    if show_ylabel:
        ax.set_ylabel("Prefill TTFT (ms)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=6)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)

    # Log-2-spaced x like the paper: equal spacing between 128..4K ticks.
    ax.set_xscale("log", base=2)
    xticks = [128, 256, 512, 1024, 2048, 4096]
    xlabels = ["128", "256", "512", "1K", "2K", "4K"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.minorticks_off()

    # Legend: monospace, with right-aligned MAPE column, header from
    # legend_title. If show_labels is True (Thor panel) include series
    # names; else (Orin) just spaces so marker+MAPE align.
    show_labels = "Thor" in legend_title
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for fw in order:
        if fw not in fits: continue
        h = Line2D([0], [0], color=COLOR[fw], marker=MARKER[fw],
                   linestyle="none", markersize=8,
                   markeredgecolor="#1f2937", markeredgewidth=0.6)
        name = LABEL[fw] if show_labels else " "
        labels.append(f"{name:<18s} {fits[fw]['mape']:>5.1f}%")
        handles.append(h)
    leg = ax.legend(handles, labels, title=legend_title,
                    loc="upper left", fontsize=10, frameon=True,
                    handlelength=1.2, handletextpad=0.6,
                    labelspacing=0.35, prop={"family": "monospace", "size": 10},
                    title_fontsize=10)
    leg.get_title().set_fontweight("bold")

    # Formula box in the upper-right of the Orin panel.
    if show_formula_box:
        ax.text(0.98, 0.98,
                r"$L_{\mathrm{pref}}(I) = a\,I_p^{2} + b\,I_p + c$"
                "\n"
                r"$I_p = \lceil I/128 \rceil \cdot 128$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#d1d5db", lw=0.8, alpha=0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    thor_rows = load_rows(SWEEP_THOR)
    orin_rows = load_rows(SWEEP_ORIN)
    # Merge the Orin pytorch_compile side-sweep into orin_rows.
    for fw, pts in load_rows(COMPILE_ORIN).items():
        orin_rows[fw].extend(pts)
    # Orin llama.cpp FA-on TTFT ships separately (absent from the baseline sweep)
    FA_ORIN = SWEEP_ORIN.parent / "llamacpp_fa_orin.csv"
    if FA_ORIN.exists():
        for r in csv.DictReader(open(FA_ORIN)):
            I = _fnum(r.get("prompt_tokens")); t = _fnum(r.get("ttft_ms"))
            if I is not None and t is not None and I <= 4096:
                orin_rows["llamacpp_fa"].append((int(I), t))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 5.6))
    draw_panel(axL, thor_rows, ORDER_THOR,
               title="AGX Thor 128 GB (LPDDR5X 273 GB/s)",
               legend_title="Thor Fit MAPE",
               show_ylabel=True, show_formula_box=False)
    draw_panel(axR, orin_rows, ORDER_ORIN,
               title="AGX Orin 32 GB (LPDDR5 204.8 GB/s)",
               legend_title="Orin Fit MAPE",
               show_ylabel=False, show_formula_box=True)

    fig.tight_layout()
    pdf = out_dir / "fig05_ttft_fits.pdf"
    png = out_dir / "fig05_ttft_fits.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
