#!/usr/bin/env python3
"""Shared fit + data-loading helpers for the analytical performance-model figures
(Fig 5 TTFT, Fig 6 TPOT, Fig 8 energy).

De-hardcoded: reads the shipped sweep CSVs directly (no embedded DATA dicts). The
fit math is ported verbatim from the original generator so coefficients are identical.

Column mapping (validated against the original hardcoded DATA_16BIT):
  I        = prompt_tokens
  O        = gen_tokens
  ttft     = ttft_ms          (exact match to original)
  tpot     = tpot_ms          (exact match to original)
  dec_w    = (dec_total_mw + dec_dram_mw) / 1000     # 4-rail decode power, W
  e_tok_mj = dec_w * tpot                            # per-token 4-rail decode energy, mJ
For Thor the DRAM rail is merged (dec_dram_mw = 0), so dec_w = dec_total_mw/1000.
"""
import csv, math
import numpy as np
from statistics import mean

# fp16 / 16-bit aliases treated as the "16-bit" curve.
# NOTE: fp16_nocache / fp16_mb32 are vLLM cache/batch ablation variants — NOT the
# reference fp16 cell — so they are deliberately excluded (folding them in mixes two
# populations and wrecks the vLLM fit).
FP16 = {"fp16", "bf16", "f16", "16-bit"}

# framework list + display per platform (llama.cpp uses FA-on on Thor)
CANON = {
    "orin": ["trtllm", "vllm", "sglang", "llamacpp", "pytorch"],
    "thor": ["trtedge_llm", "vllm", "sglang", "llamacpp_fa", "pytorch"],
}
FW_LABEL = {
    "trtllm": "TensorRT-LLM", "trtedge_llm": "TensorRT-Edge",
    "vllm": "vLLM", "sglang": "SGLang",
    "llamacpp": "llama.cpp", "llamacpp_fa": "llama.cpp (FA)",
    "pytorch": "PyTorch",
}
FW_COLOR = {
    "trtllm": "#1f77b4", "trtedge_llm": "#1f77b4",
    "vllm": "#ff7f0e", "sglang": "#9467bd",
    "llamacpp": "#e377c2", "llamacpp_fa": "#be185d", "pytorch": "#2ca02c",
}
FW_MARKER = {
    "trtllm": "o", "trtedge_llm": "o", "vllm": "s", "sglang": "^",
    "llamacpp": "D", "llamacpp_fa": "*", "pytorch": "v",
}


def sweep_csv(repo, plat):
    """The single 15-run sweep per platform (raw N-rows/cell or pre-averaged;
    the loaders average per cell either way)."""
    from pathlib import Path
    name = "sweep_locked_thor.csv" if plat == "thor" else "sweep_locked.csv"
    return Path(repo) / "data" / "chat" / name


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def i_pad(i):
    """128-aligned padded input length (Tensor-Core block size)."""
    return int(math.ceil(i / 128.0) * 128)


def lsq(X, y):
    if len(y) < X.shape[1]:
        return None
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def mape(measured, predicted):
    pairs = [(m, p) for m, p in zip(measured, predicted) if m]
    if not pairs:
        return None
    return 100.0 * mean(abs(p - m) / m for m, p in pairs)


def load_fp16_rows(csv_path, model="Llama-3.2-1B"):
    """Read a shipped 62-col sweep CSV -> {raw_fw: [row dicts]} for fp16 cells of
    the given model. row = {fw, I, O, ttft, tpot, dec_w, e_tok_mj}.

    The sweep CSVs may contain several models (e.g. Orin ships Llama-3.2-1B +
    Llama-3.1-8B rows); the fits are Llama-3.2-1B, so filter by the `model` column."""
    out = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("quantization") or "").strip() not in FP16:
                continue
            if model and (r.get("model") or "").strip() != model:
                continue
            fw = r.get("framework")
            I, O = fnum(r.get("prompt_tokens")), fnum(r.get("gen_tokens"))
            if I is None or O is None:
                continue
            ttft, tpot = fnum(r.get("ttft_ms")), fnum(r.get("tpot_ms"))
            dt = fnum(r.get("dec_total_mw"))
            dd = fnum(r.get("dec_dram_mw")) or 0.0
            dec_w = (dt + dd) / 1000.0 if dt is not None else None
            e_tok = dec_w * tpot if (dec_w is not None and tpot is not None) else None
            out.setdefault(fw, []).append(
                {"fw": fw, "I": int(I), "O": int(O),
                 "ttft": ttft, "tpot": tpot, "dec_w": dec_w, "e_tok_mj": e_tok})
    # Average per (fw, I, O) cell — robust to raw N-rows/cell OR pre-averaged 1-row/cell.
    import collections
    from statistics import fmean
    def _m(g, k):
        v = [x[k] for x in g if x[k] is not None]
        return fmean(v) if v else None
    for fw in list(out):
        by = collections.OrderedDict()
        for r in out[fw]:
            by.setdefault((r["I"], r["O"]), []).append(r)
        out[fw] = [{"fw": fw, "I": I, "O": O, "ttft": _m(g, "ttft"),
                    "tpot": _m(g, "tpot"), "dec_w": _m(g, "dec_w"), "e_tok_mj": _m(g, "e_tok_mj")}
                   for (I, O), g in by.items()]
    return out


# ----------------------- fits (ported verbatim) ---------------------
def fit_prefill(rows_fw):
    """L_pref(I) = a I_pad^2 + b I_pad + c."""
    by_I = {}
    for r in rows_fw:
        if r["ttft"] is None:
            continue
        by_I.setdefault(r["I"], []).append(r["ttft"])
    Is = sorted(by_I)
    if len(Is) < 3:
        return None
    Ip = np.array([i_pad(i) for i in Is], dtype=float)
    Tm = np.array([mean(by_I[i]) for i in Is], dtype=float)
    coef = lsq(np.column_stack([Ip * Ip, Ip, np.ones_like(Ip)]), Tm)
    if coef is None:
        return None
    a, b, c = coef
    pred = a * Ip * Ip + b * Ip + c
    return {"a": a, "b": b, "c": c, "Is": Is, "Tm": Tm, "pred": pred, "mape": mape(Tm, pred)}


def fit_decode(rows_fw):
    """TPOT(I,O) = m*(I + (O-1)/2) + n."""
    pts = [(r["I"] + (r["O"] - 1) / 2.0, r["tpot"]) for r in rows_fw if r["tpot"] is not None]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    coef = lsq(np.column_stack([xs, np.ones_like(xs)]), ys)
    if coef is None:
        return None
    m, n = coef
    return {"m": m, "n": n, "xs": xs, "ys": ys, "pred": m * xs + n, "mape": mape(ys, m * xs + n)}


def fit_energy(rows_fw):
    """E_tok(O) = alpha ln(O) + beta (mJ/tok)."""
    by_O = {}
    for r in rows_fw:
        if r["e_tok_mj"] is None:
            continue
        by_O.setdefault(r["O"], []).append(r["e_tok_mj"])
    Os = sorted(by_O)
    if len(Os) < 2:
        return None
    Oa = np.array(Os, dtype=float)
    Em = np.array([mean(by_O[o]) for o in Os], dtype=float)
    coef = lsq(np.column_stack([np.log(Oa), np.ones_like(Oa)]), Em)
    if coef is None:
        return None
    alpha, beta = coef
    return {"alpha": alpha, "beta": beta, "Os": Os, "Em": Em,
            "pred": alpha * np.log(Oa) + beta, "mape": mape(Em, alpha * np.log(Oa) + beta)}


def setup_axes(ax, xlabel, ylabel, title=None, logx=False, logy=False):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
