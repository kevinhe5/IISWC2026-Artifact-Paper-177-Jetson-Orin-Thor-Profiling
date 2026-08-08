#!/usr/bin/env python3
"""Fig 5 — Prefill TTFT vs input length, Llama-3.2-1B fp16 (AGX Orin | AGX Thor).
De-hardcoded: reads the shipped 15-run sweep CSVs; no embedded DATA.

  python3 gen_fig05_ttft_fits.py [--out DIR]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fits_common import (load_fp16_rows, fit_prefill, i_pad, setup_axes,
                          sweep_csv, CANON, FW_LABEL, FW_COLOR, FW_MARKER)

REPO = Path(__file__).resolve().parents[2]
TITLE = {"orin": "AGX Orin 32 GB", "thor": "AGX Thor 128 GB"}


def panel(ax, plat):
    rows = load_fp16_rows(sweep_csv(REPO, plat))
    for fw in CANON[plat]:
        f = fit_prefill(rows.get(fw, []))
        if f is None:
            continue
        Ip = np.array([i_pad(i) for i in f["Is"]], dtype=float)
        ax.scatter(Ip, f["Tm"], color=FW_COLOR[fw], marker=FW_MARKER[fw],
                   label=FW_LABEL[fw], s=32, zorder=3)
        g = np.linspace(Ip.min(), Ip.max(), 200)
        ax.plot(g, f["a"] * g * g + f["b"] * g + f["c"], color=FW_COLOR[fw],
                ls="--", lw=1.2, alpha=0.8, zorder=2)
    setup_axes(ax, r"Input sequence length $I$ (tokens)", "Prefill TTFT (ms)",
               title=TITLE[plat])
    ax.legend(fontsize=8, loc="upper left", frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, plat in zip(axes, ("orin", "thor")):
        panel(ax, plat)
    fig.tight_layout()
    out = Path(a.out) / "fig05_ttft_fits"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
