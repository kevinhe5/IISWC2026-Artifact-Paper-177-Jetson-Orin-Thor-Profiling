#!/usr/bin/env python3
"""RECONSTRUCTED generator for fig_v5_roofline_pp2048_orin_thor.{pdf,png}
(paper \\label{fig:roofline}, texts/v5_02_landscape.tex,
 caption "Roofline Model of different frameworks at ISL=2k OSL=2k").

Original generator lost in the Thor wipe. Rebuilt from the sibling
gen_fig_v5_roofline_combined.py (which overlays TWO platform ceilings but
only plots Orin data at pp=512). This version:
  * plots at pp = 2048 (ISL=2k),
  * plots BOTH devices: AGX Orin (FILLED markers) + AGX Thor (HOLLOW markers),
  * reads achieved TFLOPS straight from the surviving sweep CSVs
    (dec_tflops / pp_tflops columns), so every framework point is data-driven.

Log-log roofline. Each framework contributes up to four points per device:
  fp16 decode  (star, AI=1)      fp16 prefill  (star, AI=pp=2048)
  4-bit decode (square, AI~3.5-4) 4-bit prefill (square, AI=pp*AI_dec)
X = arithmetic intensity (FLOPs/byte); Y = achieved TFLOPS (median over repeats).

Platform ceilings (Table tab:platforms, sibling roofline scripts):
  AGX Orin 32 GB : FP16 dense 50 TFLOPS (=100 sparse/2), INT4 ~400 TOPS, BW 204.8 GB/s
  AGX Thor 128 GB: FP16 518 TFLOPS, INT4 2070 TOPS, BW 273 GB/s
draw_roofline() reproduced from gen_fig_v5_roofline_combined.py.

Data: Orin  data/chat/sweep_locked.csv
      Thor  data/chat/sweep_locked_thor.csv
Llama-3.2-1B, pp=2048, locked clocks, MAXN.
"""
from __future__ import annotations
import csv
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse as _argparse
_AP = _argparse.ArgumentParser(); _AP.add_argument("--out", default="."); _ARGS, _ = _AP.parse_known_args()
_REPO = Path(__file__).resolve().parents[2]
OUT_BASE = Path(_ARGS.out) / "fig02_roofline"
ORIN_CSV = str(_REPO / "data" / "chat" / "sweep_locked.csv")            # 15-run sweep
THOR_CSV = str(_REPO / "data" / "chat" / "sweep_locked_thor.csv")       # 15-run sweep
PP = 2048

FW_COLOR = {
    "TRT-LLM":   "#1f77b4",
    "vLLM":      "#ff7f0e",
    "SGLang":    "#9467bd",
    "llama.cpp": "#e377c2",
    "PyTorch":   "#2ca02c",
}

# Canonical framework -> (csv framework name, fp16 quant, 4bit quant, AI_decode_4bit)
# AI_decode: fp16 = 1.0 (bpp=2); int4/NF4 = 4.0 (bpp=0.5); Q4_0/gguf ~ 3.55.
# AI_prefill = pp * AI_decode.
DEVSPEC = {
    "Orin": {
        "csv": ORIN_CSV,
        "fw": {
            "TRT-LLM":   ("trtllm",   "fp16",         "int4",        4.0),
            "vLLM":      ("vllm",      "fp16_nocache", "gguf_Q4_0",   3.55),
            "SGLang":    ("sglang",    "fp16_nocache", None,          None),
            "llama.cpp": ("llamacpp",  "f16",          "Q4_0",        3.55),
            "PyTorch":   ("pytorch",   "bf16",         "4bit",        4.0),
        },
    },
    "Thor": {
        "csv": THOR_CSV,
        "fw": {
            "TRT-LLM":   ("trtedge_llm", "fp16", "int4_awq",     4.0),
            "vLLM":      ("vllm",        "fp16", "gguf_Q4_0",    3.55),
            "SGLang":    ("sglang",      "fp16", None,           None),
            "llama.cpp": ("llamacpp",    "f16",  "Q4_0",         3.55),
            "PyTorch":   ("pytorch",     "bf16", "4bit",         4.0),
        },
    },
}

# label, peak_FP16_TFLOPS, peak_BW_GB_s, color, peak_INT4_TOPS
PLATFORMS = [
    ("AGX Orin 32 GB",   50.0,  204.8, "#1f77b4",  400.0),
    ("AGX Thor 128 GB", 518.0,  273.0, "#d62728", 2070.0),
]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def median_tflops(rows, fwname, quant, col):
    """Median of `col` over rows matching (framework, quant, pp=PP) with col>0."""
    vals = [_num(r[col]) for r in rows
            if r["framework"] == fwname and r["quantization"] == quant
            and r["prompt_tokens"] == str(PP)]
    vals = [v for v in vals if v and v > 0]
    return statistics.median(vals) if vals else None


def collect():
    """Return list of (device, fw, quant_class, phase, AI, TFLOPS)."""
    pts = []
    for dev, spec in DEVSPEC.items():
        rows = load(spec["csv"])
        for fw, (fwname, q16, q4, ai4) in spec["fw"].items():
            # fp16 decode  (AI=1) and prefill (AI=pp)
            if q16:
                d = median_tflops(rows, fwname, q16, "dec_tflops")
                if d:
                    pts.append((dev, fw, "fp16", "decode", 1.0, d))
                p = median_tflops(rows, fwname, q16, "pp_tflops")
                if p:
                    pts.append((dev, fw, "fp16", "prefill", float(PP), p))
            # 4-bit decode (AI=ai4) and prefill (AI=pp*ai4)
            if q4:
                d = median_tflops(rows, fwname, q4, "dec_tflops")
                if d:
                    pts.append((dev, fw, "4bit", "decode", ai4, d))
                p = median_tflops(rows, fwname, q4, "pp_tflops")
                if p:
                    pts.append((dev, fw, "4bit", "prefill", PP * ai4, p))
    return pts


def draw_roofline(ax, peak_fp16, peak_int4, peak_bw_gbps, x_min, x_max, color):
    bw = peak_bw_gbps / 1000.0
    knee_fp16 = peak_fp16 / bw
    knee_int4 = peak_int4 / bw
    ax.plot([x_min, knee_int4], [bw * x_min, bw * knee_int4],
            color=color, lw=1.4, ls="--", alpha=0.85)
    ax.plot([knee_fp16, x_max], [peak_fp16, peak_fp16],
            color=color, lw=1.6, ls="-", alpha=0.85)
    ax.plot([knee_fp16], [peak_fp16], marker="x", color=color, ms=8, mew=2)
    ax.plot([knee_int4, x_max], [peak_int4, peak_int4],
            color=color, lw=1.2, ls=":", alpha=0.55)
    ax.plot([knee_int4], [peak_int4], marker="x", color=color, ms=8, mew=1.5, alpha=0.7)


def main():
    x_min, x_max = 0.3, 2e4
    y_min, y_max = 3e-2, 5e3
    pts = collect()

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    ceiling_handles = []
    for label, pf16, pbw, color, pint4 in PLATFORMS:
        draw_roofline(ax, pf16, pint4, pbw, x_min, x_max, color)
        ceiling_handles.append(plt.Line2D([0], [0], color=color, lw=1.6, ls="--",
                                          label=f"{label}"))

    # Framework points: fp16 -> star, 4bit -> square; Orin filled, Thor hollow.
    for dev, fw, qc, phase, ai, tf in pts:
        marker = "*" if qc == "fp16" else "s"
        ms = 16 if qc == "fp16" else 9
        c = FW_COLOR[fw]
        if dev == "Orin":
            ax.plot([ai], [tf], marker=marker, ms=ms, mew=0.9,
                    mfc=c, mec="#0a0a0e", ls="")
        else:  # Thor hollow
            ax.plot([ai], [tf], marker=marker, ms=ms, mew=1.6,
                    mfc="none", mec=c, ls="")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("arithmetic intensity  (FLOPs / byte)", fontsize=11)
    ax.set_ylabel("achieved TFLOPS", fontsize=11)
    ax.grid(True, which="both", color="#e5e7eb", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#374151", labelsize=10)

    ax.text(1.0, y_max * 0.4, "decode", fontsize=12, fontweight="bold",
            color="#374151", ha="center")
    ax.text(PP, y_max * 0.4, "prefill", fontsize=12, fontweight="bold",
            color="#374151", ha="center")

    # Legend: marker-class + device-fill + per-framework color + 2 ceilings.
    style_handles = [
        plt.Line2D([0], [0], marker="*", ls="", ms=13, mfc="#888",
                   mec="#0a0a0e", label="fp16"),
        plt.Line2D([0], [0], marker="s", ls="", ms=8, mfc="#888",
                   mec="#0a0a0e", label="4-bit"),
        plt.Line2D([0], [0], marker="*", ls="", ms=13, mfc="#888",
                   mec="#0a0a0e", label="Orin (filled)"),
        plt.Line2D([0], [0], marker="*", ls="", ms=13, mfc="none",
                   mec="#888", mew=1.6, label="Thor (hollow)"),
    ]
    fw_handles = [plt.Line2D([0], [0], marker="*", ls="", mfc=FW_COLOR[f],
                             mec="#0a0a0e", ms=12, label=f)
                  for f in ["TRT-LLM", "vLLM", "SGLang", "llama.cpp", "PyTorch"]]
    ax.legend(handles=style_handles + fw_handles + ceiling_handles,
              loc="lower right", fontsize=8.5, frameon=True, ncol=2,
              framealpha=0.95, edgecolor="#d1d5db")

    fig.tight_layout(pad=0.6)
    fig.savefig(str(OUT_BASE) + ".pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(str(OUT_BASE) + ".png", bbox_inches="tight", pad_inches=0.04, dpi=200)
    print(f"wrote {OUT_BASE}.pdf / .png")
    print(f"\nRoofline points (Llama-3.2-1B, pp={PP}):")
    print(f"  {'dev':<5} {'fw':<10} {'q':<5} {'phase':<8} {'AI':>8} {'TFLOPS':>9}")
    for dev, fw, qc, phase, ai, tf in sorted(pts):
        print(f"  {dev:<5} {fw:<10} {qc:<5} {phase:<8} {ai:>8.1f} {tf:>9.3f}")


if __name__ == "__main__":
    main()
