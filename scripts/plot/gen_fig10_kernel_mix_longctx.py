#!/usr/bin/env python3
"""Generate fig_v5_fig8_kernel_mix_longctx.{pdf,png}
   — Per-framework GPU kernel-time decomposition at short (gen=128) vs
     long (gen=65 536) context, on BOTH AGX Orin and AGX Thor. Answers:
     "What grows when we stretch the decode context, and how does the
     framework architecture (and platform) amplify it?"

Layout: two stacked panels (Orin above, Thor below). Each panel renders
5 frameworks × 4 bars (Q4-short, Q4-long, fp16-short, fp16-long) with
the same bundled 5-category stack (matmul · attention · quantize ·
copy_cast · other) as fig7. Y-axes are independent so each panel's
internal kernel mix is readable; the side-by-side platform contrast is
read by comparing matching cells.

Self-contained snapshot — no runtime JSON dependency.

Source artifacts (kept for traceability, not loaded):
  Orin: data/nsys_profiles/kernel_categories.json
        (4-bit gen=128 baseline; fp16 + long-ctx values from auxiliary
        nsys runs documented in PHASE2_SUMMARY.md)
  Thor: data/nsys_profiles/kernel_categories_thor_gen128.json
        data/nsys_profiles/kernel_categories_thor_gen65536.json
        (extracted from traces_thor/{fw}_gen{128,65536}.sqlite via
        extract_kernel_categories.py --traces-dir traces_thor).

Thor short-ctx fp16 cells (trtllm/llamacpp/vllm/sglang) were not
captured in the nsys sweep — only the 4-bit default variant was traced
at gen=128 for those engines. Those cells render as "n/a" in the Thor
panel; PyTorch is the one framework that has both bf16 and NF4 short-
ctx Thor traces.

Data: the per-(fw, quant, ctx) category tables ship as
  data/nsys/kernel_categories_longctx.json        (Orin)
  data/nsys/kernel_categories_longctx_thor.json   (Thor)
loaded below as DATA_ORIN / DATA_THOR (same schema the renderer expects).
"""
from pathlib import Path

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
import argparse as _argparse
_ap = _argparse.ArgumentParser(); _ap.add_argument("--out", default=".")
_outdir = Path(_ap.parse_known_args()[0].out)
_outdir.mkdir(parents=True, exist_ok=True)
OUT_BASE = str(_outdir / "fig10_kernel_mix_longctx")

FW_ORDER  = ["trtllm", "llamacpp", "vllm", "sglang", "pytorch"]
FW_LABEL  = {"trtllm":"TRT-LLM", "llamacpp":"llama.cpp", "vllm":"vLLM",
             "sglang":"SGLang",  "pytorch":"PyTorch"}
CTX_ORDER = ["short_ctx", "long_ctx"]
CTX_LABEL = {"short_ctx":"short\n128", "long_ctx":"long\n65 K"}

CAT_BUNDLE = {
    "matmul":      ["matmul"],
    "attention":   ["attention"],
    "quantize":    ["quantize"],
    "copy_cast":   ["copy_cast"],
    # elementwise = activation/SiLU + generic elementwise + sampling. This
    # is the bulk of the "leftover compute" that used to be hidden inside
    # `other`. Pulling it out makes per-engine overhead (e.g. pytorch's
    # heavy elementwise stream) visible.
    "elementwise": ["elementwise", "activation", "sampling"],
    # other = norm + rope + kv-write. True per-step infrastructure — small
    # and roughly framework-invariant; useful as the "baseline" floor.
    "other":       ["norm", "rope", "kvcache"],
}
CAT_ORDER = ["matmul", "attention", "quantize", "copy_cast", "elementwise", "other"]
CAT_LABEL = {
    "matmul":      "matmul  (weight-stream)",
    "attention":   "attention  (KV-stream)",
    "quantize":    "quantize / dequant",
    "copy_cast":   "copy / cast",
    "elementwise": "activation + elem-wise + sampling",
    "other":       "other  (norm + rope + kv-write)",
}
# Palette matched to fig7 (gen_fig_v5_fig7_kernel_mix_quant_orin_thor) so the
# two kernel-mix figures share one legend vocabulary. The 5 fig7 categories
# use its exact hex; `elementwise` (fig8-only split out of `other`) gets a
# distinct emerald that doesn't collide with the fig7 five.
CAT_COLOR = {
    "matmul":      "#60a5fa",   # blue   (fig7)
    "attention":   "#f472b6",   # pink (paper print palette)
    "quantize":    "#f59e0b",   # amber  (fig7)
    "copy_cast":   "#a78bfa",   # violet (paper print palette)
    "elementwise": "#34d399",   # emerald (fig8-only; split from `other`)
    "other":       "#cbd5e1",   # light slate (fig7)
}

# ---------------------------------------------------------------------
# Orin Llama-3.2-1B pp=128 (Llama-3.2-1B, AGX Orin 32 GB, locked clocks).
# 5 categories, ms / decode token, gen ∈ {128, 65 536}. None = not measured.
# ---------------------------------------------------------------------
DATA_ORIN = json.load(open(_REPO / "data/nsys/kernel_categories_longctx.json"))
# ---------------------------------------------------------------------
# Thor Llama-3.2-1B pp=128 (AGX Thor 128 GB, locked clocks).
# Extracted from traces_thor/{fw}{,_fp16}_gen{128,65536}.sqlite via
# extract_kernel_categories.py --traces-dir traces_thor.
# Thor 'TRT-LLM' = TRT-Edge-LLM 0.7.0 (NVIDIA Jetson-Thor proprietary
# build); stock TRT-LLM 0.12 unsupported on sm_121.
# Short-ctx fp16 not captured for trtllm/llamacpp/vllm/sglang.
# ---------------------------------------------------------------------
DATA_THOR = json.load(open(_REPO / "data/nsys/kernel_categories_longctx_thor.json"))

QUANT_ORDER = ["Q4", "fp16"]

# Edge color encodes platform — matches fig7_orin_thor's combined view.
PLATFORM_EDGE = {"orin": "#15803d", "thor": "#d62728"}   # Orin = dark green, Thor = red


def _lighten(hexc, amt=0.55):
    """Blend a hex color `amt` of the way toward white."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amt)
    g = int(g + (255 - g) * amt)
    b = int(b + (255 - b) * amt)
    return f"#{r:02x}{g:02x}{b:02x}"


# Lightened category fills so the blue (Orin) / red (Thor) bar EDGES are the
# visual emphasis. Edges use PLATFORM_EDGE at a heavier line width.
CAT_FILL = {k: _lighten(v, 0.55) for k, v in CAT_COLOR.items()}
EDGE_LW = 1.6


def _layout_pairs():
    """Single-panel layout. Each framework hosts 4 (quant, ctx) cells in
    order [Q4-S, Q4-L, fp16-S, fp16-L]; each cell hosts a side-by-side pair
    of bars (Orin left, Thor right).

    Returns:
      cell_centers : list[float], 4 cell-centers per framework (× 5 fw = 20)
      pair_dx      : (orin_dx, thor_dx) shift inside a pair
      bar_width    : float
      fw_mids      : list[float], per-framework x mid-point (for fw labels)
      prec_mids    : list[(q4_mid, fp_mid)] per fw (for Q4 / fp16 subtitle)
    """
    bar_width   = 0.42
    pair_dx     = 0.24
    cell_gap_in = 1.10   # between cells within a Q4/fp16 pair (short→long)
    cell_gap_q  = 1.40   # between Q4 pair and fp16 pair within a framework
    group_gap   = 2.10   # between frameworks
    cells, fw_mids, prec_mids = [], [], []
    cursor = 0.0
    for _ in FW_ORDER:
        q4_s = cursor
        q4_l = cursor + cell_gap_in
        fp_s = q4_l + cell_gap_q
        fp_l = fp_s + cell_gap_in
        cells.extend([q4_s, q4_l, fp_s, fp_l])
        fw_mids.append((q4_s + fp_l) / 2)
        prec_mids.append(((q4_s + q4_l) / 2, (fp_s + fp_l) / 2))
        cursor = fp_l + group_gap
    return (np.array(cells), (-pair_dx, +pair_dx), bar_width,
            fw_mids, prec_mids)


def _draw_one_bar(ax, x, slot, *, edge_color, bar_width, y_max, legend_dst,
                  annotate_cats=("matmul", "attention"), na_label="n/a"):
    """Render one stacked bar at x. `slot` is None or a dict like the
    DATA_* per-cell dicts. `annotate_cats` lists which stacked segments get
    an in-bar value label. `na_label` is the text shown for an empty cell
    (e.g. "OOM" for the out-of-memory case). Returns the bar-top y."""
    if not slot:
        ax.text(x, y_max * 0.015, na_label, ha="center", va="bottom",
                fontsize=9, color="#9ca3af", fontstyle="italic", rotation=90)
        return 0.0
    total = sum(slot.get(c, 0.0) for c in CAT_ORDER)
    if total <= 0:
        ax.text(x, y_max * 0.015, na_label, ha="center", va="bottom",
                fontsize=9, color="#9ca3af", fontstyle="italic", rotation=90)
        return 0.0
    bottom = 0.0
    for cat in CAT_ORDER:
        v = slot.get(cat, 0.0)
        if v <= 0: continue
        lbl = CAT_LABEL[cat] if cat not in legend_dst else None
        h = ax.bar(x, v, width=bar_width, bottom=bottom,
                   color=CAT_FILL[cat],
                   edgecolor=edge_color, linewidth=EDGE_LW,
                   zorder=3, label=lbl)
        if cat not in legend_dst: legend_dst[cat] = h
        bottom += v
    return bottom


def _draw_combined(ax, data_orin, data_thor):
    """One wide panel: 5 fw × 4 (quant,ctx) cells × 2 platforms = 40 bars.
    Each (fw, quant, ctx) cell hosts a side-by-side Orin+Thor pair."""
    cells, (dx_orin, dx_thor), bar_width, fw_mids, prec_mids = _layout_pairs()

    # Y range from union of both platforms.
    def _bar_total(d, fw, quant, ctx):
        slot = d.get(fw, {}).get(quant, {}).get(ctx) if d.get(fw, {}).get(quant) else None
        return sum(slot.get(c, 0.0) for c in CAT_ORDER) if slot else 0.0
    totals = []
    for fw in FW_ORDER:
        for q in QUANT_ORDER:
            for ctx in CTX_ORDER:
                totals.append(_bar_total(data_orin, fw, q, ctx))
                totals.append(_bar_total(data_thor, fw, q, ctx))
    y_max = max(totals) * 1.06   # tight headroom; legend sits above the panel

    legend_dst = {}
    idx = 0
    for fw in FW_ORDER:
        for q in QUANT_ORDER:
            for ctx in CTX_ORDER:
                xc = cells[idx]; idx += 1
                slot_o = data_orin.get(fw, {}).get(q, {}).get(ctx) if data_orin.get(fw, {}).get(q) else None
                slot_t = data_thor.get(fw, {}).get(q, {}).get(ctx) if data_thor.get(fw, {}).get(q) else None
                # PyTorch's copy_cast (KV-concat) is a headline cost, so label
                # it too; other engines only get matmul + attention.
                ann = (("matmul", "attention", "copy_cast")
                       if fw == "pytorch" else ("matmul", "attention"))
                # Top panel shows only the per-bar TOTAL on top; per-category
                # numbers live on the delta subplot below. The one empty cell
                # (Orin PyTorch fp16 long) is an out-of-memory failure.
                na_lbl = ("OOM" if (fw == "pytorch" and q == "fp16"
                                    and ctx == "long_ctx") else "n/a")
                top_o = _draw_one_bar(ax, xc + dx_orin, slot_o,
                                      edge_color=PLATFORM_EDGE["orin"],
                                      bar_width=bar_width, y_max=y_max,
                                      legend_dst=legend_dst, annotate_cats=ann,
                                      na_label=na_lbl)
                top_t = _draw_one_bar(ax, xc + dx_thor, slot_t,
                                      edge_color=PLATFORM_EDGE["thor"],
                                      bar_width=bar_width, y_max=y_max,
                                      legend_dst=legend_dst, annotate_cats=ann,
                                      na_label=na_lbl)
                if slot_o and top_o > 0:
                    ax.text(xc + dx_orin, top_o + y_max * 0.008,
                            f"{top_o:.1f}", ha="center", va="bottom",
                            fontsize=14, color=PLATFORM_EDGE["orin"],
                            fontweight="bold")
                if slot_t and top_t > 0:
                    ax.text(xc + dx_thor, top_t + y_max * 0.008,
                            f"{top_t:.1f}", ha="center", va="bottom",
                            fontsize=14, color=PLATFORM_EDGE["thor"],
                            fontweight="bold")

    # Label rows under the axis, top→bottom:
    #   1. O / T per bar (platform; blue/red)
    #   2. short / long per cell
    #   3. Q4 / fp16 per quant-pair
    ax.set_xticks([])
    xtrans = ax.get_xaxis_transform()
    # Row 1 — O / T markers (first, just under the axis).
    for xc in cells:
        ax.text(xc + dx_orin, -0.035, "O", transform=xtrans, ha="center",
                va="top", fontsize=16, color=PLATFORM_EDGE["orin"],
                fontweight="bold")
        ax.text(xc + dx_thor, -0.035, "T", transform=xtrans, ha="center",
                va="top", fontsize=16, color=PLATFORM_EDGE["thor"],
                fontweight="bold")
    # Row 2 — short / long per cell (below O/T).
    ctx_lbl = ["short", "long", "short", "long"]
    for i, xc in enumerate(cells):
        ax.text(xc, -0.105, ctx_lbl[i % 4], transform=xtrans, ha="center",
                va="top", fontsize=15, color="#1f2937")
    # Row 3 — Q4 / fp16 subtitle.
    for (q4_mid, fp_mid) in prec_mids:
        ax.text(q4_mid, -0.175, "Q4", transform=xtrans, ha="center",
                va="top", fontsize=19, color="#4b5563", fontstyle="italic")
        ax.text(fp_mid, -0.175, "fp16", transform=xtrans, ha="center",
                va="top", fontsize=19, color="#4b5563", fontstyle="italic")

    # Framework labels are omitted on the top panel — the aligned bottom
    # panel carries them, and the columns line up.

    # Vertical framework separators.
    for i in range(len(fw_mids) - 1):
        sep = (fw_mids[i] + fw_mids[i + 1]) / 2
        ax.axvline(sep, color="#d4d4d8", linewidth=0.8, linestyle="--",
                   zorder=1, alpha=0.7)

    ax.set_ylim(0, y_max)
    ax.set_xlim(cells[0] - 1.0, cells[-1] + 1.0)   # shared with delta panel
    ax.set_ylabel("GPU kernel time\n(ms / decode token)",
                  fontsize=22, labelpad=8)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.5, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=19)

    return legend_dst


def _draw_short_long_delta(ax, data_orin, data_thor):
    """Lower panel: per-category ms/tok ADDED going short(128)→long(65K),
    stacked. 4 bars per framework — Q4(O,T) and fp16(O,T) — mirroring the
    top panel. Shows the long-context OVERHEAD: attention (KV-stream)
    dominates everywhere; matmul ≈ 0 except llama.cpp-fp16-Thor; PyTorch
    copy_cast spikes. Orin = blue edge, Thor = red edge."""
    # Aligned to the TOP panel: reuse its _layout_pairs() x-coords. Each
    # framework's Q4 delta-pair is centered under the top's Q4 (S,L) cells,
    # fp16 delta-pair under its fp16 cells. Each bar is the S→L overhead.
    # Light fills + heavy blue/red edges (Orin/Thor). One bar per (quant,
    # platform); the two it summarizes are the S and L cells directly above.
    cells, (dx_orin, dx_thor), bar_width, fw_mids, prec_mids = _layout_pairs()

    def _delta(data, fw, quant):
        cell = data.get(fw, {}).get(quant, {})
        s, l = cell.get("short_ctx"), cell.get("long_ctx")
        if not s or not l:
            return None
        return {c: l.get(c, 0.0) - s.get(c, 0.0) for c in CAT_ORDER}

    def _bar(x, d, plat, na_label="n/a"):
        if d is None:
            ax.text(x, 0.4, na_label, ha="center", va="bottom", rotation=90,
                    fontsize=9, color="#9ca3af", fontstyle="italic"); return
        bottom = 0.0
        for cat in CAT_ORDER:
            v = max(d[cat], 0.0)
            if v <= 0.02: continue
            ax.bar(x, v, width=bar_width, bottom=bottom, color=CAT_FILL[cat],
                   edgecolor=PLATFORM_EDGE[plat], linewidth=EDGE_LW, zorder=3)
            # Per-category value labels live only on this (delta) subplot,
            # and only for the three categories the text argues about.
            if cat in ("matmul", "attention", "copy_cast") and v >= 1.5:
                ax.text(x, bottom + v / 2, f"{v:.1f}", ha="center", va="center",
                        rotation=90, fontsize=8, color="#0f1115",
                        fontweight="bold", zorder=6)
            bottom += v
        if bottom > 0:
            ax.text(x, bottom + 0.6, f"+{bottom:.0f}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=PLATFORM_EDGE[plat])

    for fi, fw in enumerate(FW_ORDER):
        q4_mid, fp_mid = prec_mids[fi]
        # PyTorch fp16(bf16) long OOMs on Orin, so its S→L delta is undefined.
        na = "OOM" if fw == "pytorch" else "n/a"
        _bar(q4_mid + dx_orin, _delta(data_orin, fw, "Q4"),   "orin")
        _bar(q4_mid + dx_thor, _delta(data_thor, fw, "Q4"),   "thor")
        _bar(fp_mid + dx_orin, _delta(data_orin, fw, "fp16"), "orin", na_label=na)
        _bar(fp_mid + dx_thor, _delta(data_thor, fw, "fp16"), "thor", na_label=na)

    ax.set_ylabel("long-context overhead\n(Δ ms/tok, S→L)", fontsize=22, labelpad=8)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.5, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim(cells[0] - 1.0, cells[-1] + 1.0)   # match top panel exactly
    xt = ax.get_xaxis_transform()
    for fi, fw in enumerate(FW_ORDER):
        q4_mid, fp_mid = prec_mids[fi]
        # "S→L" marker under each delta bar (which top S/L pair it came from)
        # One centered "S→L" per quant pair (platform is already shown by the
        # yellow/red bar edges) — avoids the two-label overlap.
        for mid in (q4_mid, fp_mid):
            ax.text(mid, -0.05, "S→L", transform=xt, ha="center",
                    va="top", fontsize=16, fontweight="bold", color="#374151")
        ax.text(q4_mid, -0.13, "Q4", transform=xt, ha="center", va="top",
                fontsize=15, color="#4b5563", fontstyle="italic")
        ax.text(fp_mid, -0.13, "fp16", transform=xt, ha="center", va="top",
                fontsize=15, color="#4b5563", fontstyle="italic")
        ax.text(fw_mids[fi], -0.22, FW_LABEL[fw], transform=xt, ha="center",
                va="top", fontsize=20, fontweight="bold", color="#1f2937")
    for i in range(len(fw_mids) - 1):
        ax.axvline((fw_mids[i] + fw_mids[i + 1]) / 2, color="#d4d4d8",
                   lw=0.8, ls="--", alpha=0.7, zorder=1)


def main():
    # Two stacked panels:
    #   top    — full stacked kernel mix (5 fw × 4 cells × 2 platforms)
    #   bottom — zoomed grouped per-category overhead at long context (log y)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(18.0, 9.4),
        gridspec_kw=dict(height_ratios=[1.0, 0.52], hspace=0.34))

    legend_dst = _draw_combined(ax, DATA_ORIN, DATA_THOR)
    _draw_short_long_delta(ax2, DATA_ORIN, DATA_THOR)

    # Legend floats in the empty band above the (short) vLLM / SGLang bars,
    # horizontally centered. Semi-transparent so any bar peeking through
    # stays readable.
    from matplotlib.patches import Patch
    cat_handles = [legend_dst[c] for c in CAT_ORDER if c in legend_dst]
    plat_handles = [
        Patch(facecolor="#e5e7eb", edgecolor=PLATFORM_EDGE["orin"],
              linewidth=2.0, label="Orin (O — dark green edge)"),
        Patch(facecolor="#e5e7eb", edgecolor=PLATFORM_EDGE["thor"],
              linewidth=2.0, label="Thor (T — red edge)"),
    ]
    # Horizontal 2-row band ABOVE the top panel (8 entries / 4 cols = 2 rows),
    # so it never sits over the bars and the full plot height is usable.
    ax.legend(handles=cat_handles + plat_handles,
              loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=4, fontsize=18, frameon=True, framealpha=0.95,
              edgecolor="#d1d5db", borderpad=0.5,
              handlelength=1.4, handleheight=1.1, columnspacing=1.8,
              labelspacing=0.5).set_zorder(10)

    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.07)

    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", pad_inches=0.18, dpi=200)
    print(f"wrote {OUT_BASE}.pdf")
    print(f"wrote {OUT_BASE}.png")


if __name__ == "__main__":
    main()
