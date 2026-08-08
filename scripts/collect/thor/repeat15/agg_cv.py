#!/usr/bin/env python3
"""Aggregate σ (absolute std) and CV distributions across cells, per metric — Thor.
Matches the Orin summary format: for each metric report the across-cell distribution of the
per-cell σ AND per-cell CV (mean/median/min/max/sd/p90)."""
import glob, json, os, statistics as st

D = os.environ.get("PROFILE_ROOT", "/nvme/iiswc/Jetson_profile") + "/data/benchmarks/thor/data/repeat_stats"

def load(f): return [json.loads(l) for l in open(f) if l.strip()]
def get(r, path):
    cur = r
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur: return None
        cur = cur[k]
    try: return float(cur)
    except: return None
def p90(xs): xs = sorted(xs); return xs[min(int(len(xs)*0.9), len(xs)-1)]
def dist(xs):
    return dict(mean=st.mean(xs), median=st.median(xs), min=min(xs), max=max(xs),
                sd=st.pstdev(xs), p90=p90(xs), n=len(xs))

# metric: (json key, unit-divisor, unit, cell-filter)
#   cell filter: 'all' = every cell; 'prefill' = only cells with a reliable (nonzero, ISL>=512) prefill sample
def cells_for(kind):
    fs = glob.glob(f"{D}/*_fp16_pp*_gen*.jsonl")
    # exclude the pytorch_compile compile-cost TPOT outlier cell for TPOT only
    return fs

METRICS = [
    ("TPOT",           "tpot_ms",              1.0,   "ms", "decode"),
    ("TTFT",           "ttft_ms",              1.0,   "ms", "all"),
    ("Prefill power",  "prefill_power.total_mw",1000.,"W",  "prefill"),
    ("Prefill energy", "prefill_energy_mj",    1000., "J",  "prefill"),
    ("Decode power",   "decode_power.total_mw", 1000.,"W",  "decode"),
    ("Decode energy",  "decode_energy_mj",     1000., "J",  "decode"),
]

def isl_of(fname):  # pp<ISL>_gen
    import re; m = re.search(r"pp(\d+)_gen", fname); return int(m.group(1)) if m else 0

for name, key, div, unit, filt in METRICS:
    sigmas, cvs, EXC = [], [], []
    for f in cells_for(name):
        base = os.path.basename(f)
        rows = load(f)
        v = [get(r, key) for r in rows]; v = [x/div for x in v if x is not None]
        if len(v) < 2: continue
        m = st.mean(v)
        if m <= 0: continue
        # compile-cost artifact: pytorch_compile TPOT dominated by a one-time torch.compile
        # dynamic-recompile that num_runs=3 didn't amortize on short decode. Steady-state TPOT
        # is <20ms (matches master sweep); contaminated draws are >50ms. Excluded from CV headline.
        if name == "TPOT" and base.startswith("pytorch_compile") and m > 50:
            EXC.append(base); continue
        # prefill power/energy: only trust cells with ISL>=512 (short prefill = too few power samples)
        if filt == "prefill":
            tt=[get(r,"ttft_ms") for r in rows]; tt=[x for x in tt if x]
            if not tt or st.mean(tt) < 200: continue  # prefill must last >~10 tegrastats samples (21ms)
        sigmas.append(st.pstdev(v)); cvs.append(st.pstdev(v)/m*100)
    if not sigmas: print(f"{name}: no cells"); continue
    S, C = dist(sigmas), dist(cvs)
    print(f"\n{name} σ (across {S['n']} cells): mean {S['mean']:.3f} {unit}, median {S['median']:.3f} {unit}, "
          f"min {S['min']:.3f} {unit}, max {S['max']:.3f} {unit}, sd {S['sd']:.3f} {unit}, p90 {S['p90']:.3f} {unit};")
    print(f"  CV of mean {C['mean']:.2f} %, median {C['median']:.2f} %, min {C['min']:.2f} %, "
          f"max {C['max']:.2f} %, sd {C['sd']:.2f} %, p90 {C['p90']:.2f} %.")
    if EXC:
        print(f"  [excluded {len(EXC)} compile-cost cell(s): {', '.join(sorted(set(EXC)))}]")
