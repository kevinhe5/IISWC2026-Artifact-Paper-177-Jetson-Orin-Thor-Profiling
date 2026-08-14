#!/usr/bin/env python3
"""Fig 8 — per-token energy fits, two panels: (left) prefill energy per prompt
token vs input length I with fit E(I) = c/I + d + eI; (right) decode energy per
output token vs output length O with fit E(O) = mO + n. Formatting follows the
paper figure (fig_v5_energy_fits_combined.pdf): per-panel monospace legend with
a right-aligned "Fit MAPE" column, formula boxes, solid fit curves, dark-edged
markers, log-x with 128/256/512/1K/2K/4K ticks.

Data: data/chat/sweep_locked.csv (Llama-3.2-1B fp16-class rows; energy columns
prefill_energy_mj / decode_energy_mj averaged per cell over the raw runs). The
long-context decode rows (gen > 4096) come from data/chat/longctx_fp16_orin.csv.

  python3 gen_fig08_energy_fits.py [--out DIR]
"""
import argparse, csv
from pathlib import Path
from collections import defaultdict
from statistics import fmean, mean
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/chat/sweep_locked.csv"
FP16 = {"fp16", "bf16", "f16"}   # + fp16_nocache for vLLM/SGLang (canonical cells)

# (csv fw token, paper display label) — display labels as printed in the paper
SERIES = [
    ("trtllm",   "TRT-Edge-LLM"),
    ("llamacpp", "llama.cpp"),
    ("vllm",     "vLLM"),
    ("sglang",   "SGLang"),
    ("pytorch",  "PyTorch (eager)"),
]
FW_COLOR = {"trtllm": "#1f77b4", "llamacpp": "#e377c2", "vllm": "#ff7f0e",
            "sglang": "#9467bd", "pytorch": "#2ca02c"}
FW_MARKER = {"trtllm": "o", "llamacpp": "D", "vllm": "s", "sglang": "^",
             "pytorch": "v"}


def load():
    """fw -> {"pre": {I: [mJ/prompt-tok]}, "dec": {O: [mJ/output-tok]}}"""
    out = {fw: {"pre": defaultdict(list), "dec": defaultdict(list)}
           for fw, _ in SERIES}
    cell = defaultdict(list)
    for r in csv.DictReader(open(SRC)):
        if (r.get("model") or "Llama-3.2-1B") != "Llama-3.2-1B":
            continue
        fw = r["framework"]
        if fw not in out:
            continue
        q = (r.get("quantization") or "").strip()
        if q not in FP16 and q != "fp16_nocache":
            continue
        if fw in ("vllm", "sglang") and q != "fp16_nocache" and q != "fp16":
            continue
        try:
            I = int(r["prompt_tokens"]); O = int(r["gen_tokens"])
            pe = float(r["prefill_energy_mj"]); de = float(r["decode_energy_mj"])
        except (TypeError, ValueError):
            continue
        cell[(fw, q, I, O)].append((pe, de))
    # average per cell (raw N rows), prefer the no-cache variant where present
    have_nc = {(fw) for (fw, q, _, _) in cell if q == "fp16_nocache"}
    for (fw, q, I, O), v in cell.items():
        if fw in have_nc and q != "fp16_nocache":
            continue
        pe = fmean(p for p, _ in v); de = fmean(d for _, d in v)
        if pe > 0:
            out[fw]["pre"][I].append(pe / I)
        if de > 0:
            out[fw]["dec"][O].append(de / O)
    # long-context decode rows (gen > 4096) ship separately (e_tok already mJ/tok)
    LONG = SRC.parent / "longctx_fp16_orin.csv"
    if LONG.exists():
        for r in csv.DictReader(open(LONG)):
            fw = r["framework"]
            if fw not in out:
                continue
            try:
                O = int(r["gen_tokens"]); e = float(r["e_tok_mj"])
            except (TypeError, ValueError):
                continue
            out[fw]["dec"][O].append(e)
    return out


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 100.0 * np.mean(np.abs(p - y) / y)


def fit_prefill(pts):
    """E(I) = c/I + d + e*I"""
    Is = np.array(sorted(pts)); ys = np.array([mean(pts[i]) for i in Is])
    X = np.column_stack([1.0 / Is, np.ones_like(Is, dtype=float), Is.astype(float)])
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    pred = X @ coef
    return Is, ys, coef, mape(ys, pred)


def fit_decode(pts):
    """E(O) = m*O + n"""
    Os = np.array(sorted(pts)); ys = np.array([mean(pts[o]) for o in Os])
    X = np.column_stack([Os.astype(float), np.ones_like(Os, dtype=float)])
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    pred = X @ coef
    return Os, ys, coef, mape(ys, pred)


def ticks(ax, vals):
    ax.set_xscale("log")
    ax.set_xticks(vals)
    ax.set_xticklabels(["131K" if v == 131072 else (f"{v//1024}K" if v >= 1024 else str(v)) for v in vals],
                       fontsize=11)
    ax.minorticks_off()


def legend_mono(ax, entries, header, loc="upper left"):
    """Monospace legend with a right-aligned MAPE column (paper style)."""
    from matplotlib.lines import Line2D
    handles, labels = [], []
    handles.append(Line2D([], [], ls="", label=""))
    labels.append(f"{'':<18}{header:>8}")
    for fw, disp, m in entries:
        handles.append(Line2D([], [], marker=FW_MARKER[fw], ls="",
                              mfc=FW_COLOR[fw], mec="#0f1115", ms=7))
        labels.append(f"{disp:<18}{m:>6.0f}%")
    leg = ax.legend(handles, labels, loc=loc, fontsize=9.5, frameon=True,
                    framealpha=0.95, edgecolor="#d1d5db", labelspacing=0.45,
                    handlelength=1.1, handletextpad=0.6,
                    prop={"family": "monospace", "size": 9.5})
    return leg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    data = load()

    fig, (axp, axd) = plt.subplots(1, 2, figsize=(10.6, 4.4))

    pre_entries, dec_entries = [], []
    for fw, disp in SERIES:
        # ---- prefill panel ----
        if data[fw]["pre"]:
            Is, ys, coef, m = fit_prefill(data[fw]["pre"])
            axp.scatter(Is, ys, color=FW_COLOR[fw], marker=FW_MARKER[fw],
                        s=42, edgecolor="#0f1115", linewidth=0.7, zorder=4)
            Ig = np.geomspace(Is.min(), Is.max(), 200)
            axp.plot(Ig, coef[0] / Ig + coef[1] + coef[2] * Ig,
                     color=FW_COLOR[fw], lw=1.8, zorder=3)
            pre_entries.append((fw, disp, m))
        # ---- decode panel ----
        if data[fw]["dec"]:
            Os, ys, coef, m = fit_decode(data[fw]["dec"])
            axd.scatter(Os, ys, color=FW_COLOR[fw], marker=FW_MARKER[fw],
                        s=42, edgecolor="#0f1115", linewidth=0.7, zorder=4)
            Og = np.geomspace(Os.min(), Os.max(), 200)
            axd.plot(Og, coef[0] * Og + coef[1],
                     color=FW_COLOR[fw], lw=1.8, zorder=3)
            dec_entries.append((fw, disp, m))

    for ax, title, xlab, ylab in (
            (axp, "Prefill phase", "Input prompt length $I$ (tokens, log scale)",
             "Prefill energy / prompt token (mJ)"),
            (axd, "Decode phase", "Output length $O$ (tokens, log scale)",
             "Decode energy / output token (mJ)")):
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.set_xlabel(xlab, fontsize=12)
        ax.set_ylabel(ylab, fontsize=12)
        ax.grid(True, which="both", color="#e5e7eb", lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=11)
    ticks(axp, [128, 256, 512, 1024, 2048, 4096])
    ticks(axd, [128, 512, 2048, 8192, 32768, 131072])

    legend_mono(axp, pre_entries, "Fit MAPE", loc="upper left")
    legend_mono(axd, dec_entries, "Fit MAPE", loc="upper left")

    box = dict(boxstyle="round,pad=0.35", facecolor="#f8fafc",
               edgecolor="#94a3b8", lw=1.0)
    axp.text(0.04, 0.06, r"$E(I) = c/I + d + eI$", transform=axp.transAxes,
             fontsize=11, bbox=box)
    axd.text(0.96, 0.80, r"$E(O) = mO + n$", transform=axd.transAxes,
             fontsize=11, ha="right", va="top", bbox=box)

    fig.tight_layout(pad=0.8)
    out = Path(a.out) / "fig08_energy_fits"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.1, dpi=200)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
