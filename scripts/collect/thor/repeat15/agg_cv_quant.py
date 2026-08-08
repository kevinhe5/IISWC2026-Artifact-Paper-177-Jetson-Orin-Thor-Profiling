#!/usr/bin/env python3
"""Per-(framework,quant) CV aggregation over the repeat_stats cells — includes power/energy rails.
Groups every repeat_stats/*.jsonl by framework+quant and reports, per group, the across-cell
distribution of per-cell CV for the reviewer-#1 metrics. Same guards as agg_cv.py:
 - prefill power/energy only from cells with a reliable prefill (mean TTFT >= 200 ms)
 - pytorch_compile TPOT compile-cost cells (mean > 50 ms) excluded."""
import glob, json, os, re, statistics as st

D = os.environ.get("PROFILE_ROOT", "/nvme/iiswc/Jetson_profile") + "/data/benchmarks/thor/data/repeat_stats"
METRICS = [  # display, json key, divisor, unit, filter
    ("TPOT",        "tpot_ms",               1.0,   "ms", "decode"),
    ("TTFT",        "ttft_ms",               1.0,   "ms", "all"),
    ("Dec power",   "decode_power.total_mw", 1000., "W",  "decode"),
    ("Dec energy",  "decode_energy_mj",      1000., "J",  "decode"),
    ("Pre power",   "prefill_power.total_mw",1000., "W",  "prefill"),
    ("Pre energy",  "prefill_energy_mj",     1000., "J",  "prefill"),
]
QRE = re.compile(r"^(?P<fw>.+?)_(?P<quant>gguf_Q[0-9A-Za-z_]+|Q[0-9][0-9A-Za-z_]*|fp16|f16|bf16|4bit|8bit)_pp(\d+)_gen(\d+)$")

def load(f): return [json.loads(l) for l in open(f) if l.strip()]
def get(r, path):
    cur = r
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur: return None
        cur = cur[k]
    try: return float(cur)
    except: return None

# collect cells -> group (fw,quant)
groups = {}
for f in glob.glob(f"{D}/*_pp*_gen*.jsonl"):
    m = QRE.match(os.path.basename(f)[:-6])
    if not m: continue
    groups.setdefault((m['fw'], m['quant']), []).append(f)

def cell_cv(f, key, div, filt):
    rows = load(f)
    v = [get(r, key) for r in rows]; v = [x/div for x in v if x is not None]
    if len(v) < 2: return None
    mean = st.mean(v)
    if mean <= 0: return None
    if filt == "prefill":
        tt = [get(r, "ttft_ms") for r in rows]; tt = [x for x in tt if x]
        if not tt or st.mean(tt) < 200: return None
    return st.pstdev(v)/mean*100

print(f"{'framework':16} {'quant':12} {'cells':>5}  " + "  ".join(f"{n:>10}" for n,_,_,_,_ in METRICS))
print(f"{'':16} {'':12} {'':>5}  " + "  ".join(f"{'CVμ/max%':>10}" for _ in METRICS))
def sortkey(k): return (k[0], k[1])
for (fw, q) in sorted(groups, key=sortkey):
    fs = groups[(fw, q)]
    row = []
    for name, key, div, unit, filt in METRICS:
        cvs = []
        for f in fs:
            if name == "TPOT" and fw.startswith("pytorch_compile"):
                rows = load(f); tp = [get(r, "tpot_ms") for r in rows]; tp = [x for x in tp if x]
                if tp and st.mean(tp) > 50: continue  # compile-cost artifact
            c = cell_cv(f, key, div, filt)
            if c is not None: cvs.append(c)
        row.append(f"{st.mean(cvs):.2f}/{max(cvs):.2f}" if cvs else "-")
    print(f"{fw:16} {q:12} {len(fs):>5}  " + "  ".join(f"{c:>10}" for c in row))

# overall across every quant/variant cell (excluding fp16-only? include all)
print("\n== OVERALL (all groups pooled, per-cell CV) ==")
for name, key, div, unit, filt in METRICS:
    allcv = []
    for (fw, q), fs in groups.items():
        for f in fs:
            if name == "TPOT" and fw.startswith("pytorch_compile"):
                rows = load(f); tp = [get(r, "tpot_ms") for r in rows]; tp = [x for x in tp if x]
                if tp and st.mean(tp) > 50: continue
            c = cell_cv(f, key, div, filt)
            if c is not None: allcv.append(c)
    if allcv:
        print(f"  {name:11} CV mean {st.mean(allcv):.2f}% / median {st.median(allcv):.2f}% / max {max(allcv):.2f}%  (n={len(allcv)} cells)")
