#!/usr/bin/env python3
"""Fig 11 — Pareto: decode throughput (tok/s) vs energy efficiency (tok/J),
Llama-3.2-1B, pp=128/gen=128. AGX Orin (hollow markers) vs AGX Thor (filled;
label suffix "Nx" = Thor decode speedup vs the matching Orin cell).

Formatting follows the paper figure (fig_v5_figA_pareto_tps_eff_orin_thor.pdf);
data is read from the shipped repo files (values may differ from the paper's
print vintage; see METHODOLOGY_NOTES):
  Orin : data/chat/sweep_locked.csv          (decode_tps, decode_energy_mj;
         raw N rows/cell, averaged here) + data/chat/pytorch_compile.csv
  Thor : data/chat/pareto_thor/pareto_thor_base.csv (tps, tok_per_j)

  python3 gen_fig11_pareto.py [--out DIR]
"""
import argparse, csv
from pathlib import Path
from statistics import fmean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
ORIN_CSV = REPO / "data/chat/sweep_locked.csv"
ORIN_PC = REPO / "data/chat/pytorch_compile.csv"
THOR_CSV = REPO / "data/chat/pareto_thor/pareto_thor_base.csv"
GEN = 128   # tokens generated per cell: tok/J = GEN / decode_energy_mJ * 1000

# ---------------------------------------------------------------- styling
# (transcribed from the paper PDF; keep in sync with fig_v5_figA_* look)
FW_COLOR = {
    "TRT-LLM":         "#64a9f0",
    "llama.cpp":       "#f5a623",
    "vLLM":            "#b39ddb",
    "SGLang":          "#a9b7c6",
    "PyTorch":         "#e4574f",
    "PyTorch+compile": "#2e9e4f",
}
FW_MARKER = {
    "TRT-LLM":         "o",
    "llama.cpp":       "D",
    "vLLM":            "s",
    "SGLang":          "^",
    "PyTorch":         "v",
    "PyTorch+compile": "P",
}
FW_ORDER = ["TRT-LLM", "llama.cpp", "vLLM", "SGLang", "PyTorch", "PyTorch+compile"]
MS = {"o": 210, "D": 170, "s": 200, "^": 220, "v": 220, "P": 260}
EDGE = "#14161a"
LABEL_C = "#374151"
LEADER_C = "#9ca3af"

# ------------------------------------------------------- cell selections
# (framework, csv fw token, csv quant token, display label) — the paper's set
ORIN_CELLS = [
    ("TRT-LLM", "trtllm",   "fp16",         "fp16"),
    ("TRT-LLM", "trtllm",   "int8",         "int8"),
    ("TRT-LLM", "trtllm",   "int4",         "int4 (W4A16)"),
    ("llama.cpp", "llamacpp", "f16",        "gguf f16"),
    ("llama.cpp", "llamacpp", "Q8_0",       "gguf Q8_0"),
    ("llama.cpp", "llamacpp", "Q4_K_M",     "gguf Q4_K_M"),
    ("llama.cpp", "llamacpp", "Q4_0",       "gguf Q4_0"),
    ("vLLM", "vllm",       "fp16",         "fp16"),
    ("vLLM", "vllm",       "gguf_Q8_0",    "gguf Q8_0"),
    ("vLLM", "vllm",       "gguf_Q4_K_M",  "gguf Q4_K_M"),
    ("vLLM", "vllm",       "gguf_Q4_0",    "gguf Q4_0"),
    ("SGLang", "sglang",   "fp16",         "fp16"),
    ("PyTorch", "pytorch", "bf16",         "bf16 (eager)"),
    ("PyTorch", "pytorch", "8bit",         "bnb-int8"),
    ("PyTorch", "pytorch", "4bit",         "bnb-NF4"),
]
THOR_CELLS = [
    ("TRT-LLM", "trtllm",   "fp16",        "fp16"),
    ("TRT-LLM", "trtllm",   "int8_sq",     "int8_sq"),
    ("TRT-LLM", "trtllm",   "int4_awq",    "int4_awq"),
    ("TRT-LLM", "trtllm",   "fp8",         "fp8"),
    ("TRT-LLM", "trtllm",   "nvfp4",       "nvfp4"),
    ("llama.cpp", "llamacpp", "f16",       "gguf f16"),
    ("llama.cpp", "llamacpp", "Q8_0",      "gguf Q8_0"),
    ("llama.cpp", "llamacpp", "Q4_K_M",    "gguf Q4_K_M"),
    ("llama.cpp", "llamacpp", "Q4_0",      "gguf Q4_0"),
    ("vLLM", "vllm",       "fp16",        "fp16"),
    ("vLLM", "vllm",       "gguf_Q8_0",   "gguf Q8_0"),
    ("vLLM", "vllm",       "gguf_Q4_K_M", "gguf Q4_K_M"),
    ("vLLM", "vllm",       "gguf_Q4_0",   "gguf Q4_0"),
    ("SGLang", "sglang",   "fp16",        "fp16"),
    ("SGLang", "sglang",   "fp8",         "fp8"),
    ("SGLang", "sglang",   "gguf_Q8_0",   "gguf Q8_0"),
    ("SGLang", "sglang",   "gguf_Q4_K_M", "gguf Q4_K_M"),
    ("SGLang", "sglang",   "gguf_Q4_0",   "gguf Q4_0"),
    ("PyTorch", "pytorch", "bf16_eager",  "bf16 (eager)"),
    ("PyTorch", "pytorch", "bnb_int8",    "bnb-int8"),
    ("PyTorch", "pytorch", "bnb_nf4",     "bnb-NF4"),
    ("PyTorch+compile", "pytorch_compile", "bf16", "bf16"),
]
# Hand-tuned label offsets (points) per (platform, fw_token, quant) — mirrors the
# paper's fanned-out callouts; anything far from its point gets a leader line.
OFFSETS = {
    # Orin lower-left cluster
    ("orin", "vllm", "gguf_Q4_0"):    (-14, 30),
    ("orin", "vllm", "gguf_Q8_0"):    (-56, 16),
    ("orin", "vllm", "gguf_Q4_K_M"):  (-72, -6),
    ("orin", "vllm", "fp16"):         (14, 0),
    ("orin", "sglang", "fp16"):       (-46, 6),
    ("orin", "llamacpp", "f16"):      (26, -22),
    ("orin", "trtllm", "fp16"):       (-48, -2),
    ("orin", "pytorch", "bf16"):      (-30, -22),
    # Orin mid cluster (under/near the legend)
    ("orin", "llamacpp", "Q4_0"):     (12, 6),
    ("orin", "llamacpp", "Q4_K_M"):   (6, -20),
    ("orin", "trtllm", "int8"):       (12, 4),
    # Thor mid cluster (vLLM/SGLang gguf + TRT fp8/int8_sq)
    ("thor", "vllm", "gguf_Q8_0"):    (-30, 30),
    ("thor", "vllm", "gguf_Q4_K_M"):  (10, 44),
    ("thor", "sglang", "gguf_Q8_0"):  (34, -8),
    ("thor", "sglang", "gguf_Q4_K_M"): (30, -30),
    ("thor", "sglang", "fp8"):        (-38, 42),
    ("thor", "sglang", "gguf_Q4_0"):  (-20, 22),
    ("thor", "trtllm", "fp8"):        (46, -6),
    ("thor", "trtllm", "int8_sq"):    (6, -20),
    ("thor", "llamacpp", "Q4_K_M"):   (34, -42),
    # Thor lower-mid cluster
    ("thor", "pytorch", "bf16_eager"): (-16, -42),
    ("thor", "vllm", "fp16"):         (-58, 46),
    ("thor", "sglang", "fp16"):       (30, 28),
    ("thor", "pytorch_compile", "bf16"): (-16, 34),
    ("thor", "llamacpp", "f16"):      (16, -8),
    ("thor", "pytorch", "bnb_nf4"):   (12, 8),
    ("thor", "pytorch", "bnb_int8"):  (10, 8),
    ("thor", "trtllm", "fp16"):       (36, 6),
    ("thor", "vllm", "gguf_Q4_0"):    (-54, 16),
    ("orin", "pytorch", "4bit"):      (-42, 4),
}

# Thor cell -> Orin cell whose decode tps anchors the "Nx" speedup suffix
THOR_TO_ORIN = {
    ("trtllm", "fp16"): ("trtllm", "fp16"),
    ("trtllm", "int8_sq"): ("trtllm", "int8"),
    ("trtllm", "int4_awq"): ("trtllm", "int4"),
    ("llamacpp", "f16"): ("llamacpp", "f16"),
    ("llamacpp", "Q8_0"): ("llamacpp", "Q8_0"),
    ("llamacpp", "Q4_K_M"): ("llamacpp", "Q4_K_M"),
    ("llamacpp", "Q4_0"): ("llamacpp", "Q4_0"),
    ("vllm", "fp16"): ("vllm", "fp16"),
    ("vllm", "gguf_Q8_0"): ("vllm", "gguf_Q8_0"),
    ("vllm", "gguf_Q4_K_M"): ("vllm", "gguf_Q4_K_M"),
    ("vllm", "gguf_Q4_0"): ("vllm", "gguf_Q4_0"),
    ("sglang", "fp16"): ("sglang", "fp16"),
    ("pytorch", "bf16_eager"): ("pytorch", "bf16"),
    ("pytorch", "bnb_int8"): ("pytorch", "8bit"),
    ("pytorch", "bnb_nf4"): ("pytorch", "4bit"),
    ("pytorch_compile", "bf16"): ("pytorch_compile", "bf16"),
}


def _orin_cells():
    """(fw_token, quant) -> (tps, tok_per_j) from the raw sweep (mean per cell)."""
    acc = {}
    for r in csv.DictReader(open(ORIN_CSV)):
        if r["prompt_tokens"] != "128" or r["gen_tokens"] != "128":
            continue
        if (r.get("model") or "Llama-3.2-1B") != "Llama-3.2-1B":
            continue
        try:
            tps = float(r["decode_tps"]); e_mj = float(r["decode_energy_mj"])
        except (TypeError, ValueError):
            continue
        if tps <= 0 or e_mj <= 0:
            continue
        acc.setdefault((r["framework"], r["quantization"]), []).append((tps, e_mj))
    out = {}
    for k, v in acc.items():
        out[k] = (fmean(t for t, _ in v), GEN / fmean(e for _, e in v) * 1000.0)
    # vLLM/SGLang paper "fp16" = the no-cache variant when present
    for fw in ("vllm", "sglang"):
        if (fw, "fp16_nocache") in out:
            out[(fw, "fp16")] = out[(fw, "fp16_nocache")]
    # PyTorch+compile from its own CSV
    for r in csv.DictReader(open(ORIN_PC)):
        if r.get("prompt_tokens") == "128" and r.get("gen_tokens") == "128":
            try:
                tps = float(r["decode_tps"]); e_mj = float(r["decode_energy_mj"])
                out[("pytorch_compile", "bf16")] = (tps, GEN / e_mj * 1000.0)
            except (TypeError, ValueError, KeyError):
                pass
    return out


def _thor_cells():
    out = {}
    for r in csv.DictReader(open(THOR_CSV)):
        try:
            out[(r["framework"], r["quant"])] = (float(r["tps"]), float(r["tok_per_j"]))
        except (TypeError, ValueError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    orin, thor = _orin_cells(), _thor_cells()

    fig, ax = plt.subplots(figsize=(10.4, 6.4))

    def draw(fw, x, y, hollow):
        m = FW_MARKER[fw]
        if hollow:
            ax.scatter([x], [y], marker=m, s=MS[m], facecolor="white",
                       edgecolor=FW_COLOR[fw], linewidth=2.4, zorder=4)
        else:
            ax.scatter([x], [y], marker=m, s=MS[m], facecolor=FW_COLOR[fw],
                       edgecolor=EDGE, linewidth=1.5, zorder=5)

    def label(x, y, text, dx=7, dy=4):
        kw = dict(fontsize=8, color=LABEL_C, zorder=6)
        if abs(dx) > 22 or abs(dy) > 14:   # far label -> thin leader line
            ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=LEADER_C, lw=0.7,
                                        shrinkA=0, shrinkB=3), **kw)
        else:
            ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", **kw)

    # Orin (hollow)
    for fw, tok, q, disp in ORIN_CELLS:
        if (tok, q) not in orin:
            continue
        x, y = orin[(tok, q)]
        draw(fw, x, y, hollow=True)
        dx, dy = OFFSETS.get(("orin", tok, q), (7, 4))
        label(x, y, disp, dx, dy)

    # Thor (filled, with Nx suffix where an Orin anchor exists)
    for fw, tok, q, disp in THOR_CELLS:
        if (tok, q) not in thor:
            continue
        x, y = thor[(tok, q)]
        draw(fw, x, y, hollow=False)
        anchor = THOR_TO_ORIN.get((tok, q))
        if anchor and anchor in orin and orin[anchor][0] > 0:
            disp = f"{disp}  {x / orin[anchor][0]:.1f}x"
        dx, dy = OFFSETS.get(("thor", tok, q), (7, 4))
        label(x, y, disp, dx, dy)

    ax.set_xlabel("Decode throughput  (tokens / s)", fontsize=15)
    ax.set_ylabel("Energy efficiency  (tokens / J)", fontsize=15)
    ax.tick_params(labelsize=13)
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#e5e7eb", lw=0.6, alpha=0.8, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fw_handles = [Line2D([], [], marker=FW_MARKER[f], color="w",
                         markerfacecolor=FW_COLOR[f], markeredgecolor=EDGE,
                         markeredgewidth=1.3, markersize=11, label=f)
                  for f in FW_ORDER]
    leg1 = ax.legend(handles=fw_handles, loc="upper left", fontsize=11.5,
                     title="framework", title_fontsize=12.5, frameon=True,
                     framealpha=0.95, edgecolor="#d1d5db", borderpad=0.7,
                     labelspacing=0.55)
    ax.add_artist(leg1)
    plat_handles = [
        Line2D([], [], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#3b4252", markeredgewidth=2.2, markersize=12,
               label="Orin  (hollow)"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#3b4252",
               markeredgecolor=EDGE, markersize=12,
               label="Thor  (filled, Nx = decode speedup vs Orin)"),
    ]
    ax.legend(handles=plat_handles, loc="lower right", fontsize=12.5,
              title="platform", title_fontsize=13.5, frameon=True,
              framealpha=0.95, edgecolor="#d1d5db", borderpad=0.7)

    fig.tight_layout()
    out = Path(a.out) / "fig11_pareto"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.15, dpi=200)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
