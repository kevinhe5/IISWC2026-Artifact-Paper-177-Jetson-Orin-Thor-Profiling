#!/usr/bin/env python3
"""Fig 4 — per-token wall-clock decomposition (nsys), 16-bit vs 4-bit groups,
AGX Orin (green edge, O) paired with AGX Thor (red edge, T) per framework.
Formatting follows the paper figure (fig_v5_fig6_nsys_breakdown_orin_thor.pdf);
the segment math (kernel + launch + residual, profiler-bias absorption, and the
Orin 4-bit host-tax split vs the same engine's fp16 residual) is the original
gen_fig_v5_fig6_nsys_breakdown.py, rendered in its no-overhead form.

Data: data/nsys/breakdown_quant.json (see its _note for row schema/provenance).

  python3 gen_fig04_nsys_breakdown.py [--out DIR]
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/nsys/breakdown_quant.json"

COLORS = {
    "kernel":   "#BDD4E5",   # sky blue   — GPU compute
    "launch":   "#F7CB9F",   # peach      — cudaLaunchKernel + graph_launch
    "residual": "#E8B4B8",   # dusty rose — sync + memcpy + Python/host
    "gguf_tax": "#9b2226",   # crimson    — extra 4-bit host residual (GGUF V0)
}
PLAT_EDGE = {"orin": "#16a34a", "thor": "#dc2626"}
PLAT_TXT  = {"orin": "#15803d", "thor": "#dc2626"}
FW_ORDER = ["TRT-LLM", "llama.cpp", "vLLM", "SGLang", "PyTorch"]


def rows(data, fp16_res=None):
    """Original script's segment math (no-overhead form)."""
    out = {}
    for d in data:
        wall_nsys, wall_unprof, residual_raw = d[2], d[3], d[7]
        bias_total = max(wall_nsys - wall_unprof, 0.0)
        bias_absorbed = min(bias_total, residual_raw)
        residual_real = residual_raw - bias_absorbed
        if fp16_res is not None:
            base = fp16_res.get(d[0], residual_real)
            gguf_tax = max(residual_real - base, 0.0)
            base_residual = residual_real - gguf_tax
        else:
            gguf_tax = 0.0
            base_residual = residual_real
        out[d[0]] = {"label": d[0], "quant": d[1], "kernel": d[5],
                     "launch": d[6], "residual": base_residual,
                     "gguf_tax": gguf_tax}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    D = json.load(open(SRC))

    groups = {}   # (plat, qclass) -> {fw: segrow}
    for plat in ("orin", "thor"):
        r16 = rows(D[plat]["16bit"])
        fp16_res = {k: v["residual"] for k, v in r16.items()}
        # host-tax split on both platforms (vs the same engine's fp16 residual)
        r4 = rows(D[plat]["4bit"], fp16_res=fp16_res)
        groups[(plat, "16bit")] = r16
        groups[(plat, "4bit")] = r4

    # geometry: 2 quant groups x 5 fw x (O,T) pair
    bw = 0.68
    pair_gap = 0.78
    slot_gap = 2.05
    group_gap = 1.6
    bars = []          # (x, plat, qclass, fw)
    pair_ctr = []      # (x_center, fw)
    cursor = 0.0
    grp_mid = {}
    for qi, qclass in enumerate(("16bit", "4bit")):
        g0 = cursor
        for fw in FW_ORDER:
            for pi, plat in enumerate(("orin", "thor")):
                bars.append((cursor + pi * pair_gap, plat, qclass, fw))
            pair_ctr.append((cursor + pair_gap / 2, fw))
            cursor += slot_gap
        grp_mid[qclass] = (g0 + cursor - slot_gap + pair_gap) / 2
        if qi == 0:
            sep_x = cursor - slot_gap + pair_gap + (group_gap + slot_gap - pair_gap) / 2
        cursor += group_gap

    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    y_max = 55.0
    SEG_LABEL_MIN = 0.028

    for x, plat, qclass, fw in bars:
        r = groups[(plat, qclass)].get(fw)
        if not r:
            continue
        bottom = 0.0
        for key in ("kernel", "launch", "residual", "gguf_tax"):
            v = r[key]
            if v <= 0:
                continue
            ax.bar(x, v, width=bw, bottom=bottom, color=COLORS[key],
                   edgecolor=PLAT_EDGE[plat], linewidth=1.3, zorder=3)
            if v >= y_max * SEG_LABEL_MIN:
                ax.text(x, bottom + v / 2, f"{v:.1f}", ha="center", va="center",
                        fontsize=8.0, fontweight="bold",
                        color="#fafafa" if key == "gguf_tax" else "#1f2937")
            bottom += v
        ax.text(x, bottom + y_max * 0.013, f"{bottom:.1f}", ha="center",
                va="bottom", fontsize=10.5, fontweight="bold",
                color=PLAT_TXT[plat])

    # group titles + separator
    ax.text(grp_mid["16bit"], y_max * 1.06, "16-bit (fp16 / bf16)", ha="center",
            va="bottom", fontsize=17, fontweight="bold", color="#1f2937")
    ax.text(grp_mid["4bit"], y_max * 1.06, "4-bit (Q4 / int4 / NF4)", ha="center",
            va="bottom", fontsize=17, fontweight="bold", color="#1f2937")
    ax.axvline(sep_x, color="#d4d4d8", linewidth=0.9, linestyle="--",
               zorder=1, alpha=0.8)

    ax.set_ylim(0, y_max * 1.05)
    ax.set_ylabel("Per-token time (ms)", fontsize=16, labelpad=10)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.5, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=13)

    # x labels: colored O/T letters, framework names centered below each pair
    xt = [x for x, *_ in bars]
    ax.set_xticks(xt)
    ax.set_xticklabels(["O" if p == "orin" else "T" for _, p, *_ in bars],
                       fontsize=11, fontweight="bold")
    for tick, (_, plat, *_ ) in zip(ax.get_xticklabels(), bars):
        tick.set_color(PLAT_EDGE[plat])
    ax.tick_params(axis="x", length=0, pad=4)
    for cx, fw in pair_ctr:
        ax.text(cx, -0.075, fw, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=12.5, color="#111827")

    # legend (framed, upper-left area like the paper)
    handles = [
        Patch(facecolor=COLORS["kernel"], edgecolor="#6b7280", lw=0.8,
              label="GPU kernel  (compute)"),
        Patch(facecolor=COLORS["launch"], edgecolor="#6b7280", lw=0.8,
              label="launch  (cudaLaunchKernel + graph)"),
        Patch(facecolor=COLORS["residual"], edgecolor="#6b7280", lw=0.8,
              label="residual  (sync + memcpy + Python)"),
        Patch(facecolor=COLORS["gguf_tax"], edgecolor="#6b7280", lw=0.8,
              label="4-bit host tax  (GGUF V0 / bnb-NF4)"),
        Patch(facecolor="#f4f6f8", edgecolor=PLAT_EDGE["orin"], lw=2.0,
              label="Orin (O)"),
        Patch(facecolor="#f4f6f8", edgecolor=PLAT_EDGE["thor"], lw=2.0,
              label="Thor (T)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.97),
              fontsize=11.5, frameon=True, framealpha=0.95, edgecolor="#d1d5db",
              handlelength=1.4, handleheight=1.1, labelspacing=0.45)

    fig.tight_layout(pad=0.7)
    out = Path(a.out) / "fig04_nsys_breakdown"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.18, dpi=200)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
