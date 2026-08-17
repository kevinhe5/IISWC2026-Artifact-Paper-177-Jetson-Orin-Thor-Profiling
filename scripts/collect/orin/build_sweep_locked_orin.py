#!/usr/bin/env python3
"""Fold raw Orin sweep runs (master + N-rep) into the artifact-facing
`data/chat/sweep_locked.csv`.

Mirrors Thor's `build_sweep_locked_thor_15run_raw.py` in shape:
  * INPUT  master : the single-run (num_runs=1) 62-col master sweep
  * INPUT  rep15  : a directory of rep{1..N}.csv per-rep sweeps
  * OUTPUT sweep_locked.csv : concatenated RAW rows (~ N rows / cell). Plot
    loaders (`scripts/plot/_fits_common.py`) already average per cell, so we
    do not pre-average here — keeping raw preserves the CV / repeatability
    story for §B.1.1 aggregators.

Also regenerates `data/chat/mbu_pp512_gen256.csv` (Fig 3 anchor input) as
one row per (platform, framework) with the mean TPOT at pp=512, gen=256.

With --side-dir (raw side_sweeps.sh outputs) it additionally regenerates the
three side files the plots read:
  * side_compile.csv → pytorch_compile.csv   (62-col passthrough; plot
    loaders pick columns by name, so schema-superset is fine)
  * side_fa.csv      → llamacpp_fa_orin.csv  (8-col projection)
  * side_longctx.csv → longctx_fp16_orin.csv (8-col projection)
8-col projection: framework,quantization,prompt_tokens,gen_tokens,ttft_ms,
tpot_ms,dec_w,e_tok_mj with dec_w = (dec_total_mw+dec_dram_mw)/1000 (4-rail
module power) and e_tok_mj = dec_w * tpot_ms; both left empty when rails
were not captured, matching the shipped files.

Usage:
    python3 build_sweep_locked_orin.py \
        --master  DATA_ROOT/benchmarks/sweep_results/sweep_locked_master.csv \
        --rep15-dir DATA_ROOT/benchmarks/sweep_results/rep15 \
        --side-dir  DATA_ROOT/benchmarks/sweep_results \
        --out     REPO/data/chat/sweep_locked.csv \
        --mbu-out REPO/data/chat/mbu_pp512_gen256.csv \
        --compile-out REPO/data/chat/pytorch_compile.csv \
        --fa-out      REPO/data/chat/llamacpp_fa_orin.csv \
        --longctx-out REPO/data/chat/longctx_fp16_orin.csv
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def concat(master_rows, rep_rows_lists, header):
    yield from master_rows
    for rows in rep_rows_lists:
        yield from rows


def build_mbu_row(rows):
    """Aggregate to one row per (framework, quant) at pp=512 / gen=256, fp16 only.
    Emits (platform=orin, framework=<canonical>, tpot_ms=<mean>) for the Fig 3
    input. The Thor row is left to the Thor build script; we only overwrite the
    Orin rows here."""
    FP16 = {"fp16", "bf16", "f16", "16-bit"}
    LABEL = {
        "trtllm": "TRT", "vllm": "vLLM", "sglang": "SGLang",
        "llamacpp": "llama.cpp", "llamacpp_fa": "llama.cpp",
        "pytorch": "PyTorch", "pytorch_compile": "PyTorch (compile)",
    }
    acc = defaultdict(list)
    for r in rows:
        if r.get("prompt_tokens") != "512" or r.get("gen_tokens") != "256":
            continue
        if (r.get("model") or "Llama-3.2-1B") != "Llama-3.2-1B":
            continue
        if r.get("quantization") not in FP16:
            continue
        fw = r.get("framework")
        if fw not in LABEL:
            continue
        try:
            tpot = float(r["tpot_ms"])
        except (TypeError, ValueError):
            continue
        if tpot > 0:
            acc[LABEL[fw]].append(tpot)
    return {fw: statistics.fmean(vs) for fw, vs in acc.items() if vs}


SIDE_COLS = ["framework", "quantization", "prompt_tokens", "gen_tokens",
             "ttft_ms", "tpot_ms", "dec_w", "e_tok_mj"]


def project_side(src, dst):
    """62-col raw side sweep → the 8-col side schema the plots read."""
    rows, _ = read_csv(src)
    out = []
    for r in rows:
        try:
            ttft = float(r["ttft_ms"])
            tpot = float(r["tpot_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            mw = float(r.get("dec_total_mw") or 0) + float(r.get("dec_dram_mw") or 0)
        except (TypeError, ValueError):
            mw = 0.0
        dec_w = f"{mw / 1000:.2f}" if mw > 0 else ""
        e_tok = f"{mw / 1000 * tpot:.2f}" if mw > 0 else ""
        out.append({
            "framework": r["framework"], "quantization": r["quantization"],
            "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["gen_tokens"],
            "ttft_ms": f"{ttft:.2f}", "tpot_ms": f"{tpot:.3f}",
            "dec_w": dec_w, "e_tok_mj": e_tok,
        })
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIDE_COLS)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {dst} ({len(out)} rows)")


def copy_side(src, dst):
    """side_compile.csv → pytorch_compile.csv verbatim (62-col passthrough)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(Path(src).read_bytes())
    n = sum(1 for _ in open(dst)) - 1
    print(f"wrote {dst} ({n} rows, passthrough)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="master sweep CSV (num_runs=1)")
    ap.add_argument("--rep15-dir", required=True, help="directory containing rep{1..N}.csv")
    ap.add_argument("--side-dir", help="directory containing side_{compile,fa,longctx}.csv")
    ap.add_argument("--out", required=True, help="output data/chat/sweep_locked.csv")
    ap.add_argument("--mbu-out", required=True, help="output data/chat/mbu_pp512_gen256.csv")
    ap.add_argument("--compile-out", help="output data/chat/pytorch_compile.csv")
    ap.add_argument("--fa-out", help="output data/chat/llamacpp_fa_orin.csv")
    ap.add_argument("--longctx-out", help="output data/chat/longctx_fp16_orin.csv")
    a = ap.parse_args()

    # 1. Concatenate raw rows
    master_rows, header = read_csv(a.master)
    rep_lists = []
    rep_dir = Path(a.rep15_dir)
    for csv_path in sorted(rep_dir.glob("rep*.csv")):
        rows, _ = read_csv(csv_path)
        rep_lists.append(rows)
        print(f"  {csv_path.name}: {len(rows)} rows")
    total = len(master_rows) + sum(len(r) for r in rep_lists)
    print(f"  master: {len(master_rows)} rows")
    print(f"  reps  : {len(rep_lists)}")
    print(f"  total : {total} rows")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in concat(master_rows, rep_lists, header):
            w.writerow(row)
    print(f"wrote {out_path} ({total} rows)")

    # 2. MBU pp=512 gen=256 anchor — Orin rows only (Thor rows preserved if file exists)
    all_rows = list(concat(master_rows, rep_lists, header))
    orin_mbu = build_mbu_row(all_rows)

    mbu_out = Path(a.mbu_out)
    existing = {}      # (plat, fw) -> dict of columns from prior file
    fieldnames = ["platform", "framework", "tpot_ms", "source"]
    if mbu_out.exists():
        with open(mbu_out) as f:
            # The shipped file's `source` column is NOT quoted and contains
            # commas ("figC pp512/gen256 measurement (X, Y, Z)"). DictReader
            # would drop the tail under a None key; use restkey and merge
            # so we round-trip the full annotation.
            reader = csv.DictReader(f, restkey="__rest__")
            fieldnames = list(reader.fieldnames or fieldnames)
            last_col = fieldnames[-1]
            for r in reader:
                if "__rest__" in r and r["__rest__"]:
                    r[last_col] = ",".join([r[last_col]] + r.pop("__rest__"))
                elif "__rest__" in r:
                    r.pop("__rest__")
                existing[(r["platform"], r["framework"])] = r
    # Overwrite Orin rows with freshly computed tpot; keep Thor rows verbatim.
    for fw, tpot in orin_mbu.items():
        row = existing.get(("orin", fw), {"platform": "orin", "framework": fw})
        row["platform"] = "orin"
        row["framework"] = fw
        row["tpot_ms"] = f"{tpot:.4f}"
        if "source" in fieldnames:
            row["source"] = f"rebuilt from sweep_locked.csv rep-mean at pp=512/gen=256"
        existing[("orin", fw)] = row

    with open(mbu_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key in sorted(existing.keys()):
            w.writerow({k: existing[key].get(k, "") for k in fieldnames})
    print(f"wrote {mbu_out} ({len(existing)} rows)")

    # 3. Side sweeps → the three extra data/chat files (skipped if the raw
    #    file is absent, e.g. when the side stage was run with a SIDE subset)
    if a.side_dir:
        side = Path(a.side_dir)
        for src, out_arg, fn in [
            (side / "side_compile.csv", a.compile_out, copy_side),
            (side / "side_fa.csv",      a.fa_out,      project_side),
            (side / "side_longctx.csv", a.longctx_out, project_side),
        ]:
            if not out_arg:
                continue
            if not src.exists():
                print(f"  side: {src.name} not found — keeping shipped {Path(out_arg).name}")
                continue
            fn(src, Path(out_arg))


if __name__ == "__main__":
    main()
