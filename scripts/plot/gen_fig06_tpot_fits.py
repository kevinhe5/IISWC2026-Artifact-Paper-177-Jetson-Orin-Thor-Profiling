#!/usr/bin/env python3
"""Fig 6 — Decode TPOT vs effective context I + (O-1)/2, Llama-3.2-1B fp16,
TWO panels: AGX Thor (left) | AGX Orin (right), mirroring the paper figure
(fig_v5_tpot_fits_new_sweep.pdf, Overleaf HEAD): 7 series per panel, monospace
legend with a right-aligned "Thor/Orin Fit MAPE" column (series names on the
Thor panel only), TPOT(I,O) = m(I+(O-1)/2)+n formula box on the Orin panel,
SOLID linear-fit curves, log-x with 128/512/2K/8K/32K/131K ticks.

Data intake (all shipped):
    data/chat/sweep_locked_thor.csv    Thor: 7 fw incl. llamacpp{,_fa} + compile
    data/chat/sweep_locked.csv         Orin baseline (5 fw)
    data/chat/pytorch_compile.csv      Orin torch.compile side sweep
    data/chat/llamacpp_fa_orin.csv     Orin llama.cpp FA-on snapshot
    data/chat/longctx_fp16_orin.csv    Orin long-decode rows (gen > 4096)
Per-cell averaged (raw sweeps carry ~15 rows/cell). The Thor panel's x-range is
data-driven (Thor long-decode bench rows were lost in the Thor reset).

  python3 gen_fig06_tpot_fits.py [--out DIR]
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SWEEP_THOR   = REPO / "data" / "chat" / "sweep_locked_thor.csv"
SWEEP_ORIN   = REPO / "data" / "chat" / "sweep_locked.csv"
COMPILE_ORIN = REPO / "data" / "chat" / "pytorch_compile.csv"
FA_ORIN      = REPO / "data" / "chat" / "llamacpp_fa_orin.csv"
LONG_ORIN    = REPO / "data" / "chat" / "longctx_fp16_orin.csv"

FP16 = {"fp16", "bf16", "f16", "16-bit"}

ORDER_THOR = ["trtedge_llm", "vllm", "sglang", "llamacpp_fa",
              "llamacpp", "pytorch", "pytorch_compile"]
ORDER_ORIN = ["trtllm", "vllm", "sglang", "llamacpp_fa",
              "llamacpp", "pytorch", "pytorch_compile"]

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
COLOR = {
    "trtllm":          "#1f77b4",
    "trtedge_llm":     "#1f77b4",
    "vllm":            "#ff7f0e",
    "sglang":          "#9467bd",
    "llamacpp_fa":     "#22c1d1",   # cyan
    "llamacpp":        "#e377c2",   # pink
    "pytorch":         "#2ca02c",
    "pytorch_compile": "#a56a13",   # brown
}
MARKER = {
    "trtllm": "o", "trtedge_llm": "o", "vllm": "s", "sglang": "^",
    "llamacpp_fa": "D", "llamacpp": "D",
    "pytorch": "v", "pytorch_compile": "X",
}


def _fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def load_sweep(csv_path, model="Llama-3.2-1B"):
    """{fw: [(x, tpot)]} with x = I + (O-1)/2, per-cell averaged."""
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(csv_path, newline="")):
        if (r.get("quantization") or "").strip() not in FP16: continue
        if (r.get("model") or "").strip() != model: continue
        fw = r.get("framework")
        I = _fnum(r.get("prompt_tokens")); O = _fnum(r.get("gen_tokens"))
        t = _fnum(r.get("tpot_ms"))
        if I is None or O is None or t is None: continue
        # exclude pytorch_compile compile-cost cells (recompilation spikes);
        # same >50 ms rule as the repeatability aggregators
        if fw == "pytorch_compile" and t > 50.0: continue
        by[fw][I + (O - 1) / 2.0].append(t)
    return {fw: [(x, mean(ts)) for x, ts in sorted(cells.items())]
            for fw, cells in by.items()}


def load_side(csv_path, fwmap=None):
    """Side CSVs (compile / FA-on / long-context) -> {fw: [(x, tpot)]}."""
    out = defaultdict(list)
    if not csv_path.exists():
        return out
    for r in csv.DictReader(open(csv_path, newline="")):
        fw = r.get("framework")
        if fwmap:
            fw = fwmap.get(fw, fw)
        I = _fnum(r.get("prompt_tokens")); O = _fnum(r.get("gen_tokens"))
        t = _fnum(r.get("tpot_ms"))
        if I is None or O is None or t is None: continue
        out[fw].append((I + (O - 1) / 2.0, t))
    return out


def fit_tpot(pts):
    """TPOT(x) = m x + n (linear, on per-cell means)."""
    if len(pts) < 2: return None
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    coef, *_ = np.linalg.lstsq(
        np.column_stack([xs, np.ones_like(xs)]), ys, rcond=None)
    m, n = coef
    pred = m * xs + n
    mape = 100.0 * fmean(abs(p - y) / y for y, p in zip(ys, pred) if y)
    return {"m": m, "n": n, "xs": xs, "ys": ys, "mape": mape}


def draw_panel(ax, rows_by_fw, order, title, legend_title, show_ylabel,
               show_formula_box):
    fits = {}
    for fw in order:
        pts = rows_by_fw.get(fw)
        if not pts: continue
        f = fit_tpot(pts)
        if f is None: continue
        fits[fw] = f
        ax.scatter(f["xs"], f["ys"], color=COLOR[fw], marker=MARKER[fw],
                   s=42, zorder=3, edgecolor="#1f2937", linewidths=0.6)
        xg = np.geomspace(f["xs"].min(), f["xs"].max(), 200)
        ax.plot(xg, f["m"] * xg + f["n"], color=COLOR[fw], linestyle="-",
                linewidth=1.5, alpha=0.95, zorder=2)

    ax.set_xlabel(r"Effective context $I + (O-1)/2$ (tokens)", fontsize=12)
    if show_ylabel:
        ax.set_ylabel("TPOT (ms/tok)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=6)
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(labelsize=10)

    ax.set_xscale("log")
    xticks = [128, 512, 2048, 8192, 32768, 131072]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["128", "512", "2K", "8K", "32K", "131K"])
    ax.set_xlim(110, 1.6e5)
    ax.minorticks_off()

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

    if show_formula_box:
        ax.text(0.98, 0.98,
                r"$\mathrm{TPOT}(I,O) = m\,(I + (O-1)/2) + n$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#d1d5db", lw=0.8, alpha=0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    thor_rows = load_sweep(SWEEP_THOR)

    orin_rows = load_sweep(SWEEP_ORIN)
    for src, kw in ((COMPILE_ORIN, {}),
                    (FA_ORIN, {}),
                    (LONG_ORIN, {"fwmap": {"llamacpp_faon": "llamacpp_fa"}})):
        for fw, pts in load_side(src, **kw).items():
            orin_rows.setdefault(fw, []).extend(pts)

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
    pdf = out_dir / "fig06_tpot_fits.pdf"
    png = out_dir / "fig06_tpot_fits.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
