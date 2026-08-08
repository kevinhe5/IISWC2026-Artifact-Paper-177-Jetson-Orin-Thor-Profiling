#!/usr/bin/env python3
"""Fig 6 — Decode TPOT vs effective context, Llama-3.2-1B fp16 (AGX Orin | AGX Thor).
De-hardcoded: reads the shipped 15-run sweep CSVs; no embedded DATA.

  python3 gen_fig06_tpot_fits.py [--out DIR]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fits_common import (load_fp16_rows, fit_decode, setup_axes,
                          sweep_csv, CANON, FW_LABEL, FW_COLOR, FW_MARKER)

REPO = Path(__file__).resolve().parents[2]
TITLE = {"orin": "AGX Orin 32 GB", "thor": "AGX Thor 128 GB"}


def panel(ax, plat):
    rows = load_fp16_rows(sweep_csv(REPO, plat))
    for fw in CANON[plat]:
        f = fit_decode(rows.get(fw, []))
        if f is None:
            continue
        ax.scatter(f["xs"], f["ys"], color=FW_COLOR[fw], marker=FW_MARKER[fw],
                   label=FW_LABEL[fw], s=24, alpha=0.85, zorder=3)
        g = np.linspace(float(f["xs"].min()), float(f["xs"].max()), 200)
        ax.plot(g, f["m"] * g + f["n"], color=FW_COLOR[fw], ls="--", lw=1.2,
                alpha=0.8, zorder=2)
    setup_axes(ax, r"Effective context $I + (O-1)/2$ (tokens)", "TPOT (ms/tok)",
               title=TITLE[plat], logx=True)
    ax.legend(fontsize=8, loc="upper left", frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, plat in zip(axes, ("orin", "thor")):
        panel(ax, plat)
    fig.tight_layout()
    out = Path(a.out) / "fig06_tpot_fits"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
