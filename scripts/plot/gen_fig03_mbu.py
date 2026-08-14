#!/usr/bin/env python3
"""Fig 3 — 16-bit decode Memory-Bandwidth Utilization at pp=512 / gen=256,
Llama-3.2-1B, AGX Orin 32 GB vs AGX Thor 128 GB, locked clocks.

Formatting reproduced from the paper's original grouped-bar variant
(figs_original/fig_v5_figC_mbu_16bit_pp512_gen256.pdf):
per-framework grouped bars — Orin solid + Thor '///'-hatched — colored
by framework; ylabel "% of platform BW peak"; top horizontal legend with
Orin (solid swatch) and Thor (hatched swatch) entries labelled with each
platform's peak BW; percentage labels above each bar; TPOT ms text
inside each bar. Deliberately NO title / NO red memory-bound threshold /
NO footer formula (paper original strips them).

MBU is computed analytically per platform:
    BW  = (weights + KV(T)) / TPOT
    MBU = 100 · BW / peak_BW
where weights(fp16)=1.235B·2 B, KV(T)=2·16·8·64·2·T, T=pp+gen/2=640.

Framework order matches the paper: TRT-LLM, PyTorch (compile), vLLM,
llama.cpp, SGLang, PyTorch. Data intake reads BOTH platforms from
data/chat/mbu_pp512_gen256.csv (6 orin + 6 thor rows shipped).

Paper filename: fig_v5_figC_mbu_16bit_pp512_gen256.pdf

  python3 gen_fig03_mbu.py [--out DIR]
"""
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "chat" / "mbu_pp512_gen256.csv"

WEIGHTS_BYTES_FP16 = 1.235e9 * 2                # 2.470 GB
PP, GEN = 512, 256
T = PP + GEN // 2                                # 640 avg decode context
KV_PER_TOK_FP16 = 2 * 16 * 8 * 64 * 2            # 32 768 B/tok
KV_BYTES = KV_PER_TOK_FP16 * T                   # 20.97 MB
STREAM_BYTES = WEIGHTS_BYTES_FP16 + KV_BYTES

PEAK = {"orin": 204.8, "thor": 273.0}
PLATFORM_LABEL = {
    "orin": "AGX Orin 32 GB (BW 204.8 GB/s)",
    "thor": "AGX Thor 128 GB (BW 273 GB/s)",
}

# Framework display order + colors (color-picked from the paper PDF).
ORDER = ["TRT-LLM", "PyTorch (compile)", "vLLM",
         "llama.cpp", "SGLang", "PyTorch"]
# `csv_fw_to_display`: shipped CSV uses "TRT" not "TRT-LLM".
CSV_ALIAS = {"TRT": "TRT-LLM"}
COLOR = {
    "TRT-LLM":            "#60a5fa",   # blue
    "PyTorch (compile)":  "#a56a13",   # brown / tan (paper)
    "vLLM":               "#f97316",   # orange
    "llama.cpp":          "#f472b6",   # pink
    "SGLang":             "#a78bfa",   # purple
    "PyTorch":            "#34d399",   # green
}


def mbu(tpot_ms, peak_gbps):
    bw_bps = STREAM_BYTES / (tpot_ms / 1000.0)
    return 100.0 * bw_bps / (peak_gbps * 1e9)


def load_tpots():
    """Return {platform: {display_fw: tpot_ms}} from the shipped CSV."""
    tp = {"orin": {}, "thor": {}}
    for r in csv.DictReader(open(SRC)):
        plat = r["platform"]
        raw = r["framework"]
        disp = CSV_ALIAS.get(raw, raw)
        tp[plat][disp] = float(r["tpot_ms"])
    return tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="output directory")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    tp = load_tpots()

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    n = len(ORDER)
    x = np.arange(n, dtype=float)
    width = 0.38

    for i, fw in enumerate(ORDER):
        c = COLOR[fw]
        orin_t = tp["orin"].get(fw)
        thor_t = tp["thor"].get(fw)
        orin_m = mbu(orin_t, PEAK["orin"]) if orin_t is not None else 0.0
        thor_m = mbu(thor_t, PEAK["thor"]) if thor_t is not None else 0.0

        # Orin: solid coloured bar
        ax.bar(x[i] - width/2, orin_m, width=width,
               color=c, edgecolor="#0f1115", linewidth=0.8, zorder=3)
        # Thor: same colour, '///' hatch
        ax.bar(x[i] + width/2, thor_m, width=width,
               color=c, edgecolor="#0f1115", linewidth=0.8, hatch="///", zorder=3)

        # % labels above each bar
        for xi, m in ((x[i] - width/2, orin_m), (x[i] + width/2, thor_m)):
            if m <= 0: continue
            ax.text(xi, m + 1.4, f"{m:.0f}%",
                    ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color="#1f2937")

        # TPOT ms text inside each bar (positioned ~40% up so both fit
        # even for short bars).
        for xi, m, tval in ((x[i] - width/2, orin_m, orin_t),
                            (x[i] + width/2, thor_m, thor_t)):
            if tval is None or m <= 8: continue
            y = m * 0.42
            ax.text(xi, y, f"{tval:.1f} ms",
                    ha="center", va="center",
                    fontsize=10, color="#1f2937")

    ax.set_xticks(x)
    ax.set_xticklabels([fw.replace(" ", "\n") if fw == "PyTorch (compile)" else fw
                        for fw in ORDER], fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of platform BW peak", fontsize=14)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)

    # Top legend: solid swatch = Orin, hatched swatch = Thor.
    legend_handles = [
        Patch(facecolor="#60a5fa", edgecolor="#0f1115", linewidth=0.8,
              label=PLATFORM_LABEL["orin"]),
        Patch(facecolor="#60a5fa", edgecolor="#0f1115", linewidth=0.8,
              hatch="///", label=PLATFORM_LABEL["thor"]),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
               fontsize=13, handlelength=2.0, handleheight=1.4,
               columnspacing=3.0)

    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.94))
    pdf = out_dir / "fig03_mbu.pdf"
    png = out_dir / "fig03_mbu.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.08, dpi=200)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
