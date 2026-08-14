#!/usr/bin/env python3
"""Fig 9 — per-framework x per-quant GPU kernel-time decomposition, AGX Orin
(green edge) vs AGX Thor (red edge), with a lower panel showing the per-category
delta vs the fp16/bf16 baseline. Formatting follows the paper figure
(fig_v5_fig7_kernel_mix_quant_orin_thor.pdf); data comes from the shipped
category JSONs (schema identical to the original script's embedded DATA):

  Orin : data/nsys/kernel_categories_quant.json
  Thor : data/nsys/kernel_categories_quant_thor.json

  python3 gen_fig09_kernel_mix_quant.py [--out DIR]
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
ORIN_JSON = REPO / "data/nsys/kernel_categories_quant.json"
THOR_JSON = REPO / "data/nsys/kernel_categories_quant_thor.json"

FW_ORDER = ["trtllm", "llamacpp", "vllm", "sglang", "pytorch"]
FW_LABEL = {"trtllm": "TRT-LLM", "llamacpp": "llama.cpp", "vllm": "vLLM",
            "sglang": "SGLang", "pytorch": "PyTorch"}
QUANTS = ["fp16", "8bit", "4bit"]           # paper order: 16-bit, 8-bit, 4-bit
QUANT_LBL = {"fp16": "16-bit", "8bit": "8-bit", "4bit": "4-bit"}
QUANT_SHORT = {
    "trtllm":   {"fp16": "fp16",  "8bit": "int8",   "4bit": "int4"},
    "llamacpp": {"fp16": "F16",   "8bit": "Q8_0",   "4bit": "Q4_K_M"},
    "vllm":     {"fp16": "fp16",  "8bit": "Q8",     "4bit": "Q4"},
    "sglang":   {"fp16": "fp16",  "8bit": "Q8",     "4bit": "Q4"},
    "pytorch":  {"fp16": "bf16",  "8bit": "bnb-i8", "4bit": "NF4"},
}
CAT_ORDER = ["matmul", "attention", "quantize", "copy_cast", "other"]
CAT_LABEL = {
    "matmul":    "matmul  (weight-stream)",
    "attention": "attention  (KV-stream)",
    "quantize":  "quantize / dequant",
    "copy_cast": "copy / cast",
    "other":     "other  (norm + rope + kv-write)",
}
CAT_COLOR = {           # pale fills; the bar EDGE carries the platform color
    "matmul":    "#b9d0ea",
    "attention": "#f4c2c2",
    "quantize":  "#f7cf94",
    "copy_cast": "#cfc8e0",
    "other":     "#c2c9bd",
}
PLAT_EDGE = {"orin": "#16a34a", "thor": "#dc2626"}
PLAT_TXT  = {"orin": "#15803d", "thor": "#dc2626"}
PLAT_DTXT = {"orin": "#14532d", "thor": "#7f1d1d"}


def load(path):
    return json.load(open(path))


def cats_of(d, fw, q):
    slot = d.get(fw, {}).get(q)
    if not slot:
        return None
    return {c: float(slot.get(c, 0.0)) for c in CAT_ORDER}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    data = {"orin": load(ORIN_JSON), "thor": load(THOR_JSON)}

    # ---- geometry: per fw group, 3 quant slots x (O,T) pair -------------
    bw = 0.68          # bar width
    pair_gap = 0.78    # O -> T within a slot
    slot_gap = 2.05    # slot -> slot
    group_gap = 1.35   # extra between fw groups
    bars = []          # (x, plat, fw, q, cats)
    pair_ctr = {}      # (fw, q) -> x center of the O/T pair
    fw_mid = []
    seps = []
    cursor = 0.0
    for gi, fw in enumerate(FW_ORDER):
        g0 = cursor
        for q in QUANTS:
            for pi, plat in enumerate(("orin", "thor")):
                bars.append((cursor + pi * pair_gap, plat, fw, q,
                             cats_of(data[plat], fw, q)))
            pair_ctr[(fw, q)] = cursor + pair_gap / 2
            cursor += slot_gap
        fw_mid.append((g0 + cursor - slot_gap + pair_gap) / 2)
        if gi < len(FW_ORDER) - 1:
            seps.append(cursor - slot_gap + pair_gap + (group_gap + slot_gap - pair_gap) / 2)
        cursor += group_gap

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(13.4, 6.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.65, 1.0], hspace=0.06))

    totals = [sum(c.values()) for *_ , c in bars if c]
    y_max = max(totals) * 1.02
    head = y_max * 0.46          # headroom for fw titles + legend

    # ---- top panel: absolute stacked bars -------------------------------
    for x, plat, fw, q, cats in bars:
        if not cats:
            continue
        bottom = 0.0
        for cat in CAT_ORDER:
            v = cats[cat]
            if v <= 0:
                continue
            ax.bar(x, v, width=bw, bottom=bottom, color=CAT_COLOR[cat],
                   edgecolor=PLAT_EDGE[plat], linewidth=1.3, zorder=3)
            bottom += v
        ax.text(x, bottom + y_max * 0.015, f"{bottom:.1f}", ha="center",
                va="bottom", fontsize=10.5, fontweight="bold",
                color=PLAT_TXT[plat])

    ax.set_ylim(0, y_max + head)
    ax.set_ylabel("GPU kernel time\n(ms / decode token)", fontsize=13, labelpad=8)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.5, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.axhline(0, color="#111827", lw=1.2)

    # fw group titles
    for fw, mid in zip(FW_ORDER, fw_mid):
        ax.text(mid, y_max + head * 0.97, FW_LABEL[fw], ha="center", va="top",
                fontsize=15, fontweight="bold", color="#1f2937")

    # legend (categories + platform edges), frameless, along the top
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CAT_COLOR[c], edgecolor="#6b7280", lw=0.8,
                     label=CAT_LABEL[c]) for c in CAT_ORDER]
    handles += [Patch(facecolor="#eef2f7", edgecolor=PLAT_EDGE["orin"], lw=2.0,
                      label="Orin (green edge)"),
                Patch(facecolor="#eef2f7", edgecolor=PLAT_EDGE["thor"], lw=2.0,
                      label="Thor (red edge)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, 0.90),
              fontsize=10.5, frameon=False, ncol=4, columnspacing=1.3,
              handlelength=1.3, handleheight=1.05, labelspacing=0.35)

    # ---- bottom panel: delta vs fp16/bf16 -------------------------------
    d_lo, d_hi = 0.0, 0.0
    for x, plat, fw, q, cats in bars:
        if not cats or q == "fp16":
            continue
        base = cats_of(data[plat], fw, "fp16")
        if not base:
            continue
        deltas = {c: cats[c] - base[c] for c in CAT_ORDER}
        pos = 0.0
        neg = 0.0
        for cat in CAT_ORDER:
            v = deltas[cat]
            if v == 0:
                continue
            if v > 0:
                axd.bar(x, v, width=bw, bottom=pos, color=CAT_COLOR[cat],
                        edgecolor=PLAT_EDGE[plat], linewidth=1.1, zorder=3)
                pos += v
            else:
                axd.bar(x, v, width=bw, bottom=neg, color=CAT_COLOR[cat],
                        edgecolor=PLAT_EDGE[plat], linewidth=1.1, zorder=3)
                neg += v
        net = sum(deltas.values())
        cap_y = neg if net < 0 else pos
        axd.plot([x - bw / 2, x + bw / 2], [net, net], color="#111827",
                 lw=2.0, zorder=4)
        if net < 0:
            axd.text(x, neg - 0.35, f"{net:.1f}", ha="center", va="top",
                     fontsize=8.5, fontweight="bold", color=PLAT_DTXT[plat],
                     rotation=90)
        else:
            axd.text(x + (-0.08 if plat == "orin" else 0.08), pos + 0.35,
                     f"+{net:.1f}", ha="right" if plat == "orin" else "left",
                     va="bottom", fontsize=8.5, fontweight="bold", color="#7c2d12")
        d_lo = min(d_lo, neg)
        d_hi = max(d_hi, pos)

    axd.axhline(0, color="#111827", lw=1.2)
    axd.set_ylim(d_lo * 1.30, d_hi + 2.2)
    axd.set_ylabel("$\\Delta$ vs fp16/bf16\n(ms / decode token)", fontsize=13,
                   labelpad=8)
    axd.grid(True, axis="y", color="#e5e7eb", linewidth=0.5, alpha=0.7, zorder=1)
    axd.set_axisbelow(True)
    for sp in ("top", "right"):
        axd.spines[sp].set_visible(False)
    axd.tick_params(axis="y", labelsize=11)

    # group separators through both panels
    for s in seps:
        for a_ in (ax, axd):
            a_.axvline(s, color="#d4d4d8", linewidth=0.8, linestyle="--",
                       zorder=1, alpha=0.7)

    # ---- x labels: colored O/T per bar + two-line quant label per pair --
    xt, xl = [], []
    for x, plat, fw, q, cats in bars:
        xt.append(x); xl.append("O" if plat == "orin" else "T")
    axd.set_xticks(xt)
    axd.set_xticklabels(xl, fontsize=10, fontweight="bold")
    for tick, (x, plat, *_ ) in zip(axd.get_xticklabels(), bars):
        tick.set_color(PLAT_EDGE[plat])
    axd.tick_params(axis="x", length=0, pad=4)
    for (fw, q), cx in pair_ctr.items():
        axd.text(cx, -0.205, f"{QUANT_LBL[q]}\n{QUANT_SHORT[fw][q]}",
                 transform=axd.get_xaxis_transform(), ha="center", va="top",
                 fontsize=10.5, linespacing=1.15, color="#111827")

    fig.tight_layout(pad=0.7)
    out = Path(a.out) / "fig09_kernel_mix_quant"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.18, dpi=200)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
