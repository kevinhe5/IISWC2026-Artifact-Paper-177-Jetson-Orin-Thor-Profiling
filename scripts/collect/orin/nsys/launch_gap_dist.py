#!/usr/bin/env python3
"""Inter-kernel launch-gap distribution from an nsys sqlite.

Tier fingerprint = the MEDIAN inter-kernel gap + host/kernel ratio (architectural,
version-invariant). graph/async/V0 only change the TAIL (p99, total gap) — they
mask/expose long launch stalls but not the base launch rhythm.

For consecutive GPU kernels (sorted by start): gap_i = start[i+1] - end[i]
(clamped >=0). Reports median, p99, total gap, host/kernel = totalgap/kernelbusy.

Usage: launch_gap_dist.py <trace.sqlite> [label]
"""
import sqlite3, sys, statistics

def analyze(sqlite_path, label=""):
    c = sqlite3.connect(sqlite_path).cursor()
    # kernel activity table name varies slightly across nsys versions
    tbl = None
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL",):
        try:
            c.execute(f"SELECT COUNT(*) FROM {t}"); tbl = t; break
        except sqlite3.OperationalError:
            pass
    if not tbl:
        print(f"{label}: no kernel table"); return None
    rows = c.execute(f"SELECT start, end FROM {tbl} ORDER BY start").fetchall()
    rows = [(s, e) for s, e in rows if s is not None and e is not None]
    if len(rows) < 3:
        print(f"{label}: too few kernels ({len(rows)})"); return None
    kernel_busy = sum(e - s for s, e in rows)
    gaps = []
    for i in range(len(rows) - 1):
        g = rows[i + 1][0] - rows[i][1]
        if g >= 0:
            gaps.append(g)
    gaps_sorted = sorted(gaps)
    med = statistics.median(gaps_sorted)
    p99 = gaps_sorted[int(0.99 * len(gaps_sorted))]
    total_gap = sum(gaps)
    hk = total_gap / kernel_busy if kernel_busy else float("nan")
    def fmt_ns(x):
        return f"{x:.0f} ns" if x < 1000 else (f"{x/1000:.1f} µs" if x < 1e6 else f"{x/1e6:.1f} ms")
    print(f"{label:34} n_kern={len(rows):>6} host/kern={hk:5.2f} "
          f"gap_median={fmt_ns(med):>9} gap_p99={fmt_ns(p99):>9} total_gap={total_gap/1e6:8.1f} ms")
    return {"label": label, "n_kernels": len(rows), "host_kern": hk,
            "gap_median_ns": med, "gap_p99_ns": p99, "total_gap_ms": total_gap / 1e6}

if __name__ == "__main__":
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
