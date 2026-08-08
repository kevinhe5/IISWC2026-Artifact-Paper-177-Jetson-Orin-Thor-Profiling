#!/usr/bin/env python3
"""Fig6 per-token time breakdown for Thor, from decode-only nsys sqlite traces.

The Thor decode traces are captured with --capture-range=cudaProfilerApi
(cudaProfilerStart/Stop wraps exactly the GEN decode steps), so the whole trace
window == pure decode; no NVTX prefill/full_gen subtraction is needed. Mirrors
orin extract_breakdown.py's metrics: wall / kernel / launch / sync / memcpy /
graph_launch / other_api / residual, per decode token.

Usage: breakdown_thor.py OUT.json fw=path.sqlite[:gen] [fw=... ]
(fw label 'trtedge_llm' etc.; gen defaults to 128)
"""
import json, sqlite3, sys
from pathlib import Path


def measure(sql, framework, gen=128):
    c = sqlite3.connect(str(sql)).cursor()
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def has(t): return t in tables

    starts, ends = [], []
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if has(t):
            r = c.execute(f"SELECT MIN(start), MAX(end) FROM {t}").fetchone()
            if r and r[0] is not None:
                starts.append(r[0]); ends.append(r[1])
    wall_ns = (max(ends) - min(starts)) if starts else 0

    if has("CUPTI_ACTIVITY_KIND_KERNEL"):
        kernel_ns, n_kernels = c.execute(
            "SELECT COALESCE(SUM(end-start),0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    else:
        kernel_ns, n_kernels = 0, 0

    if has("CUPTI_ACTIVITY_KIND_MEMCPY"):
        memcpy_ns, n_memcpy = c.execute(
            "SELECT COALESCE(SUM(end-start),0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()
    else:
        memcpy_ns, n_memcpy = 0, 0

    def rt(patterns):
        if not has("CUPTI_ACTIVITY_KIND_RUNTIME"):
            return 0, 0
        likes = " OR ".join(f"s.value LIKE '{p}'" for p in patterns)
        r = c.execute(f"""SELECT COALESCE(SUM(r.end-r.start),0), COUNT(*)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
            WHERE {likes}""").fetchone()
        return int(r[0] or 0), int(r[1] or 0)

    launch_ns, n_launch = rt(["cudaLaunchKernel%"])
    sync_ns, n_sync = rt(["cudaStreamSynchronize%", "cudaDeviceSynchronize%"])
    memcpy_api_ns, _ = rt(["cudaMemcpy%"])
    graph_launch_ns, n_graph = rt(["cudaGraphLaunch%"])
    if has("CUPTI_ACTIVITY_KIND_RUNTIME"):
        runtime_total_ns = int(c.execute(
            "SELECT COALESCE(SUM(end-start),0) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0] or 0)
    else:
        runtime_total_ns = 0
    other_api_ns = max(runtime_total_ns - launch_ns - sync_ns, 0)

    ms = lambda x: x / 1e6 / gen
    return {
        "framework": framework,
        "gen_tokens": gen,
        "decode_only_via_capture_range": True,
        "wall_ms_per_tok": ms(wall_ns),
        "kernel_ms_per_tok": ms(kernel_ns),
        "launch_ms_per_tok": ms(launch_ns),
        "sync_ms_per_tok": ms(sync_ns),
        "memcpy_ms_per_tok": ms(memcpy_ns),
        "memcpy_api_ms_per_tok": ms(memcpy_api_ns),
        "graph_launch_ms_per_tok": ms(graph_launch_ns),
        "other_api_ms_per_tok": ms(other_api_ns),
        "tok_per_s": (1000.0 / ms(wall_ns)) if wall_ns else 0,
        "n_kernels": int(n_kernels),
        "n_kernels_per_tok": int(n_kernels) / gen,
        "n_launch_calls": int(n_launch),
        "n_graph_launches": int(n_graph),
        "n_memcpy": int(n_memcpy),
        "n_sync_calls": int(n_sync),
        "residual_ms_per_tok": ms(max(wall_ns - kernel_ns, 0)),
    }


if __name__ == "__main__":
    out_path = sys.argv[1]
    rows = []
    for spec in sys.argv[2:]:
        fw, rest = spec.split("=", 1)
        parts = rest.split(":")
        path = parts[0]; gen = int(parts[1]) if len(parts) > 1 else 128
        r = measure(path, fw, gen)
        rows.append(r)
        print(f"{fw:<12} wall={r['wall_ms_per_tok']:6.2f} kernel={r['kernel_ms_per_tok']:6.2f} "
              f"launch={r['launch_ms_per_tok']:5.2f} sync={r['sync_ms_per_tok']:5.2f} "
              f"residual={r['residual_ms_per_tok']:6.2f} nkern/tok={r['n_kernels_per_tok']:.0f} "
              f"({r['tok_per_s']:.1f} tok/s)")
    Path(out_path).write_text(json.dumps(rows, indent=2))
    print("wrote", out_path)
