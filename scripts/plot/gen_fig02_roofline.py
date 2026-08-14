#!/usr/bin/env python3
"""Fig 2 — combined prefill+decode roofline at pp=2048, AGX Orin (filled) vs
AGX Thor (hollow), log-log. Formatting follows the paper figure
(fig_v5_roofline_pp2048_orin_thor.pdf); the plotting/ceiling code is the
original gen_fig_v5_roofline_combined.py, with the embedded DATA block replaced
by loaders over the shipped sweeps (values track the shipped data vintage):

  Orin : data/chat/sweep_locked.csv
  Thor : data/chat/sweep_locked_thor.csv

Point recipe (original "canonical formula", N = 1.236e9 no-embed params):
  decode :  TFLOPS = 2*N*decode_tps/1e12          at the pp=128/gen=128 cell
  prefill:  TFLOPS = 2*N*(pp/ttft_s)/1e12         at the pp=2048 cell
  AI     :  fp16 decode 1.0 ; Q4 decode 3.55 ; int4/NF4 decode 4.0
            prefill = pp * AI_decode  (2048 / 7276 / 8192)

  python3 gen_fig02_roofline.py [--out DIR]
"""
import argparse, csv
from pathlib import Path
from statistics import fmean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
ORIN_CSV = REPO / "data/chat/sweep_locked.csv"
THOR_CSV = REPO / "data/chat/sweep_locked_thor.csv"
PP = 2048
N_PARAMS = 1.236e9   # no-embed param count (see original script's bookkeeping note)

FW_COLORS = {
    "TRT-LLM":   "#60a5fa",
    "vLLM":      "#f97316",
    "SGLang":    "#a78bfa",
    "llama.cpp": "#f472b6",
    "PyTorch":   "#34d399",
}
FW_ORDER = ["TRT-LLM", "vLLM", "SGLang", "llama.cpp", "PyTorch"]

# (paper fw label, [candidate csv fw tokens], fp16 quant candidates,
#  4bit quant candidates, AI_decode_4bit)
SPEC = {
    "orin": [
        ("TRT-LLM",   ["trtllm"],   ["fp16"],                 ["int4"],      4.00),
        ("llama.cpp", ["llamacpp"], ["f16"],                  ["Q4_0"],      3.55),
        ("vLLM",      ["vllm"],     ["fp16_nocache", "fp16"], ["gguf_Q4_0"], 3.55),
        ("SGLang",    ["sglang"],   ["fp16_nocache", "fp16"], ["gguf_Q4_0"], 3.55),
        ("PyTorch",   ["pytorch"],  ["bf16"],                 ["4bit"],      4.00),
    ],
    "thor": [
        ("TRT-LLM",   ["trtedge_llm"],             ["fp16"], ["int4_awq"],  4.00),
        ("llama.cpp", ["llamacpp_fa", "llamacpp"], ["f16"],  ["Q4_0"],      3.55),
        ("vLLM",      ["vllm"],                    ["fp16"], ["gguf_Q4_0"], 3.55),
        ("SGLang",    ["sglang"],                  ["fp16"], ["gguf_Q4_0"], 3.55),
        ("PyTorch",   ["pytorch"],                 ["bf16"], ["4bit"],      4.00),
    ],
}

PLATFORMS = [
    # FP16 ridges = NVIDIA-published DENSE Tensor-Core peaks; INT4/FP4 ridges =
    # sparse headline numbers (see original script for derivation).
    ("AGX Orin 32 GB",    50.0, 204.8, "#1f77b4",  400.0),
    ("AGX Thor 128 GB",  518.0, 273.0, "#d62728", 2070.0),
]


def load_cells(path):
    """(fw, quant, pp, gen) -> dict of mean decode_tps / ttft_ms over raw rows."""
    acc = {}
    for r in csv.DictReader(open(path)):
        if (r.get("model") or "Llama-3.2-1B") != "Llama-3.2-1B":
            continue
        k = (r["framework"], r["quantization"], r["prompt_tokens"], r["gen_tokens"])
        try:
            acc.setdefault(k, []).append(
                (float(r["decode_tps"]), float(r["ttft_ms"])))
        except (TypeError, ValueError):
            continue
    return {k: (fmean(t for t, _ in v), fmean(x for _, x in v))
            for k, v in acc.items() if v}


def pick(cells, fws, quants, pp, gens):
    for fw in fws:
        for q in quants:
            for g in gens:
                if (fw, q, str(pp), str(g)) in cells:
                    return cells[(fw, q, str(pp), str(g))]
    return None


def collect(cells, plat):
    """[(fw_label, quant_class, AI, TFLOPS)] for one platform."""
    pts = []
    for fw, toks, q16, q4, ai4 in SPEC[plat]:
        # fp16 decode + prefill
        c = pick(cells, toks, q16, 128, [128])
        if c:
            pts.append((fw, "fp16", 1.0, 2 * N_PARAMS * c[0] / 1e12))
        c = pick(cells, toks, q16, PP, [256, 128, 512, 1024, 2048, 4096])
        if c and c[1] > 0:
            pts.append((fw, "fp16", float(PP), 2 * N_PARAMS * (PP / (c[1] / 1000.0)) / 1e12))
        # 4-bit decode + prefill
        c = pick(cells, toks, q4, 128, [128])
        if c:
            pts.append((fw, "4bit", ai4, 2 * N_PARAMS * c[0] / 1e12))
        c = pick(cells, toks, q4, PP, [256, 128, 512, 1024, 2048, 4096])
        if c and c[1] > 0:
            pts.append((fw, "4bit", PP * ai4, 2 * N_PARAMS * (PP / (c[1] / 1000.0)) / 1e12))
    return pts


def draw_roofline(ax, peak_fp16, peak_int4, peak_bw_gbps, x_min, x_max, color):
    bw_TB_per_s = peak_bw_gbps / 1000.0
    knee_fp16 = peak_fp16 / bw_TB_per_s
    knee_int4 = peak_int4 / bw_TB_per_s
    ax.plot([x_min, knee_int4], [bw_TB_per_s * x_min, bw_TB_per_s * knee_int4],
            color=color, lw=1.6, ls="--", alpha=0.85)
    ax.plot([knee_fp16, x_max], [peak_fp16, peak_fp16],
            color=color, lw=1.8, ls="-", alpha=0.85)
    ax.plot([knee_fp16], [peak_fp16], marker="x", color=color, ms=9, mew=2)
    ax.plot([knee_int4, x_max], [peak_int4, peak_int4],
            color=color, lw=1.4, ls=":", alpha=0.55)
    ax.plot([knee_int4], [peak_int4], marker="x", color=color, ms=9, mew=1.5, alpha=0.7)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    x_min, x_max = 0.3, 2e4
    y_min, y_max = 0.03, 5e3

    orin_pts = collect(load_cells(ORIN_CSV), "orin")
    thor_pts = collect(load_cells(THOR_CSV), "thor")

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    ceiling_handles = []
    for label, peak_fp16, peak_bw, color, peak_int4 in PLATFORMS:
        draw_roofline(ax, peak_fp16, peak_int4, peak_bw, x_min, x_max, color)
        ceiling_handles.append(
            plt.Line2D([0], [0], color=color, lw=1.8, ls="--", label=label))

    # Framework markers — fp16 * / 4bit square; Orin FILLED, Thor HOLLOW.
    for pts, hollow in ((orin_pts, False), (thor_pts, True)):
        for fw, qc, ai, tflops in pts:
            marker = "*" if qc == "fp16" else "s"
            ms = 18 if qc == "fp16" else 12
            if hollow:
                ax.plot([ai], [tflops], marker=marker, ms=ms, mew=1.8,
                        mfc="white", mec=FW_COLORS[fw], ls="")
            else:
                ax.plot([ai], [tflops], marker=marker, ms=ms, mew=1.0,
                        mfc=FW_COLORS[fw], mec="#0a0a0e", ls="")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("arithmetic intensity  (FLOPs / byte)", fontsize=15)
    ax.set_ylabel("achieved TFLOPS", fontsize=15)
    ax.grid(True, which="both", color="#e5e7eb", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#374151", labelsize=13)

    # phase annotations (as in the paper variant)
    ax.text(2.2, 0.9, "decode", fontsize=16, fontweight="bold", color="#374151")
    ax.text(6.0e2, 300, "prefill", fontsize=16, fontweight="bold", color="#374151")

    quant_handles = [
        plt.Line2D([0], [0], marker="*", ls="", ms=14, mfc="#888",
                   mec="#0a0a0e", label="fp16"),
        plt.Line2D([0], [0], marker="s", ls="", ms=9, mfc="#888",
                   mec="#0a0a0e", label="4 bit"),
        plt.Line2D([0], [0], marker="*", ls="", ms=14, mfc="#666",
                   mec="#0a0a0e", label="Orin (filled)"),
        plt.Line2D([0], [0], marker="*", ls="", ms=14, mfc="white",
                   mec="#666", mew=1.8, label="Thor (hollow)"),
    ]
    fw_handles = [plt.Line2D([0], [0], marker="*", ls="", mfc=FW_COLORS[f],
                             mec="#0a0a0e", ms=12, label=f) for f in FW_ORDER]
    ax.legend(handles=quant_handles + fw_handles + ceiling_handles,
              loc="lower right", fontsize=9.5, frameon=True, framealpha=0.95,
              edgecolor="#d1d5db", ncol=2)

    fig.tight_layout(pad=0.6)
    out = Path(a.out) / "fig02_roofline"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04, dpi=200)
    print(f"wrote {out}.pdf / .png")
    print(f"\nRoofline points (Llama-3.2-1B, pp={PP}):")
    for name, pts in (("Orin", orin_pts), ("Thor", thor_pts)):
        for fw, qc, ai, tf in sorted(pts):
            print(f"  {name:<5} {fw:<10} {qc:<5} AI={ai:>8.1f}  {tf:>8.3f} TFLOPS")


if __name__ == "__main__":
    main()
