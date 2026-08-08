#!/usr/bin/env python3
# Assemble the graphs-ON master sweep: iterate the eager sweep in its exact row
# order; for every vllm/sglang cell substitute the graphs-ON re-run row (matched
# by framework,quantization,prompt_tokens,gen_tokens); copy all other framework
# rows verbatim. Output is line-for-line comparable to the eager sweep.
import csv, os, sys

EAGER = os.environ.get("THOR_MASTER_SWEEP",
    os.environ.get("PROFILE_ROOT", "/nvme/iiswc/Jetson_profile")
    + "/data/benchmarks/thor/data/sweep_results_agx_thor_128gb/sweep_locked_thor_20260622.csv")
RERUN = sys.argv[1]   # vllm_sglang_cudagraph_rows.csv (header + re-run rows)
OUT   = sys.argv[2]

RERUN_FW = {"vllm", "sglang"}
KEY = lambda r: (r["framework"], r["quantization"], r["prompt_tokens"], r["gen_tokens"])

with open(EAGER) as f:
    er = csv.DictReader(f); eager_rows = list(er); fields = er.fieldnames
with open(RERUN) as f:
    rerun_rows = list(csv.DictReader(f))

rr = {}
for r in rerun_rows:
    k = KEY(r)
    if k in rr: print(f"WARN duplicate re-run cell {k}", file=sys.stderr)
    rr[k] = r

out, missing, subbed = [], [], 0
for r in eager_rows:
    if r["framework"] in RERUN_FW:
        k = KEY(r)
        if k in rr:
            out.append(rr[k]); subbed += 1
        else:
            missing.append(k); out.append(r)   # fall back to eager (should not happen)
    else:
        out.append(r)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)

print(f"total rows: {len(out)} (expected {len(eager_rows)})")
print(f"vllm/sglang substituted: {subbed}")
print(f"graph-agnostic copied: {len(out) - subbed}")
if missing:
    print(f"MISSING re-run for {len(missing)} vllm/sglang cells (kept eager):", file=sys.stderr)
    for k in missing: print("  ", k, file=sys.stderr)
# sanity: schema identical
assert fields == list(csv.reader(open(EAGER)).__next__()), "SCHEMA MISMATCH"
print("schema: 62-col identical OK" if len(fields) == 62 else f"WARN {len(fields)} cols")
