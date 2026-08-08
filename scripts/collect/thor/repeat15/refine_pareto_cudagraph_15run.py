#!/usr/bin/env python3
"""Refine the Fig 11 pareto input (pareto_thor_base.csv) with the graphs-ON 15-run means
for the vLLM/SGLang GGUF @128/128 throughput points.

The graphs-ON rows themselves are folded into the single sweep_locked_thor.csv by
build_sweep_locked_thor_15run_raw.py — there is no separate cudagraph sweep file. This
script only updates pareto_thor_base.csv, whose decode power comes from a different
pipeline (pwr_analyze pdecode_w, not the 62-col bench dec_total_mw): it keeps pareto's
native power and refines only the throughput to the 15-run mean.

Source graphs-ON rows and the repo root are env-overridable to re-run elsewhere."""
import csv, collections, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
WORK = os.environ.get("PROFILE_ROOT", "/nvme/iiswc/Jetson_profile") + "/work"
SRC = Path(os.environ.get("THOR_CUDAGRAPH_ROWS", f"{WORK}/thor_cudagraph_15run_rows.csv"))
PARETO = REPO / "data/chat/pareto_thor/pareto_thor_base.csv"

rows = list(csv.DictReader(open(SRC)))
KEY = lambda r: (r["framework"], r["quantization"], r["prompt_tokens"], r["gen_tokens"])
cells = collections.defaultdict(list)
for r in rows:
    cells[KEY(r)].append(r)


def mean(cell, col):
    return sum(float(g[col]) for g in cell) / len(cell)


# 15-run mean tps/tpot per @128/128 cell. NOTE: power differs between pipelines
# (62-col bench dec_total_mw vs pareto's pwr_analyze pdecode_w), so we KEEP pareto's
# native pdecode_w and recompute tok/J = tps_15run / pdecode_w — same power pipeline,
# only the throughput refined to the 15-run mean.
p128 = {}
for k, grp in cells.items():
    fw, q, pp, gen = k
    if pp != "128" or gen != "128":
        continue
    p128[(fw, q)] = dict(tps=mean(grp, "decode_tps"), tpot=mean(grp, "tpot_ms"))

pr = list(csv.DictReader(open(PARETO)))
pf = list(pr[0].keys())
print("consistency check (pareto tps num_runs=1 -> graphs-ON 15-run mean; power pipeline kept):")
updated = 0
for r in pr:
    key = (r["framework"], r["quant"])
    if key in p128:
        old_tps = float(r["tps"]); new_tps = p128[key]["tps"]
        pw = float(r["pdecode_w"])            # keep pareto's native power
        d = (new_tps - old_tps) / old_tps * 100
        new = dict(tps=f"{new_tps:.1f}", tpot_ms=f"{p128[key]['tpot']:.4f}",
                   tok_per_j=f"{new_tps/pw:.3f}")
        print(f"  {r['framework']:7s} {r['quant']:12s} tps {old_tps:6.1f} -> {new['tps']:>6} ({d:+.1f}%)  tok/J {r['tok_per_j']} -> {new['tok_per_j']}")
        r.update(new); updated += 1
with open(PARETO, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=pf)
    w.writeheader(); w.writerows(pr)
print(f"\nupdated {updated} pareto rows with 15-run means (vLLM/SGLang gguf @128/128); other rows unchanged")
