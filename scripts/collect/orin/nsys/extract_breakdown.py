#!/usr/bin/env python3
"""Extract per-token time breakdown from nsys sqlite traces.

For each framework's trace, compute:
  - wall_ns:        end - start of the cudaProfilerStart/Stop window
  - kernel_ns:      sum of GPU kernel durations (CUPTI_ACTIVITY_KIND_KERNEL)
  - launch_ns:      sum of cudaLaunchKernel API durations
  - sync_ns:        sum of cudaStreamSynchronize + cudaDeviceSynchronize durations
  - memcpy_ns:      sum of MEMCPY durations
  - n_kernels:      kernel launch count

Total wall = (everything that happens between profiler start/stop on any thread).
We want PER-DECODE-TOKEN times, so we divide by GEN_TOKENS = 128.

Output: JSON to stdout, also writes /work/breakdown.json
"""
import json, os, sqlite3, subprocess, sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "traces"
NSYS_BIN  = "/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nsys"
GEN_TOKENS = 128  # must match scripts/bench_*.py

def export_sqlite(rep: Path) -> Path:
    sql = rep.with_suffix(".sqlite")
    if sql.exists() and sql.stat().st_mtime >= rep.stat().st_mtime:
        return sql
    subprocess.run(
        [NSYS_BIN, "export", "--type=sqlite", "--force-overwrite=true",
         "--output=" + str(sql), str(rep)],
        check=True, capture_output=True,
    )
    return sql

def measure(rep: Path, framework: str, gen_tokens: int = GEN_TOKENS) -> dict:
    sql = export_sqlite(rep)
    conn = sqlite3.connect(str(sql))
    c = conn.cursor()

    # === NVTX-range filtering ===
    # If the bench script wrapped prefill_only and full_gen in NVTX ranges
    # (via torch.cuda.nvtx.range_push/pop), each push/pop pair shows up as
    # one row with eventType=59 (PushPop) where the `text` column holds the
    # range name and start/end are populated. Filter on text directly.
    nvtx_ranges = {}
    try:
        rows = c.execute("""
SELECT text, start, end FROM NVTX_EVENTS
WHERE text IN ('prefill_only', 'full_gen')
""").fetchall()
        for name, st, en in rows:
            if not (st and en): continue
            if name not in nvtx_ranges or (en - st) > (nvtx_ranges[name][1] - nvtx_ranges[name][0]):
                nvtx_ranges[name] = (st, en)
    except sqlite3.OperationalError:
        nvtx_ranges = {}

    has_nvtx = ('prefill_only' in nvtx_ranges and 'full_gen' in nvtx_ranges)

    # Total wall-clock = first profiled event start to last profiled event end.
    # All tables share the same nanosecond clock as nsys.
    starts, ends = [], []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"):
        try:
            row = c.execute(f"SELECT MIN(start), MAX(end) FROM {table}").fetchone()
            if row and row[0] is not None:
                starts.append(row[0]); ends.append(row[1])
        except sqlite3.OperationalError:
            pass
    wall_ns = max(ends) - min(starts) if starts else 0
    if has_nvtx:
        # Pure-decode wall: full_gen window minus 1 decode token (since
        # full_gen = P + 128*D and prefill_only = P + 1*D)
        f_st, f_en = nvtx_ranges['full_gen']
        p_st, p_en = nvtx_ranges['prefill_only']
        # full_gen wall - prefill wall difference, scaled to decode tokens
        wall_full_ns = f_en - f_st
        wall_prefill_ns = p_en - p_st
        # (wall_full - wall_prefill) covers (128*D) - (1*D) = 127*D
        wall_decode_only_ns = wall_full_ns - wall_prefill_ns
        # report wall_ns as the decode-only wall scaled to 128 tokens for chart consistency
        wall_ns_for_chart = int(wall_decode_only_ns * gen_tokens / max(gen_tokens - 1, 1))
        wall_ns = wall_ns_for_chart

    # Sum of GPU kernel durations.
    if has_nvtx:
        # Filter to pure-decode using NVTX-range subtraction.
        # full_gen window: 1 prefill + 128 decode steps = P + 128*D
        # prefill_only window: 1 prefill + 1 decode step = P + 1*D
        # Pure decode = (full - prefill) / 127, scaled to 128 = full - prefill + 1*D ≈ scale by 128/127
        f_st, f_en = nvtx_ranges['full_gen']
        p_st, p_en = nvtx_ranges['prefill_only']
        row_full = c.execute(
            f"SELECT COALESCE(SUM(end - start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
            f"WHERE start >= {f_st} AND end <= {f_en}"
        ).fetchone()
        row_pref = c.execute(
            f"SELECT COALESCE(SUM(end - start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
            f"WHERE start >= {p_st} AND end <= {p_en}"
        ).fetchone()
        full_kern_ns = int(row_full[0] or 0); full_kern_n = int(row_full[1] or 0)
        pref_kern_ns = int(row_pref[0] or 0); pref_kern_n = int(row_pref[1] or 0)
        decode_kern_ns_per_127 = max(full_kern_ns - pref_kern_ns, 0)
        decode_kern_n_per_127  = max(full_kern_n - pref_kern_n, 0)
        # Scale to 128 decode tokens = (full - prefill) * 128 / 127
        kernel_ns = int(decode_kern_ns_per_127 * gen_tokens / max(gen_tokens - 1, 1))
        n_kernels = int(decode_kern_n_per_127 * gen_tokens / max(gen_tokens - 1, 1))
    else:
        row = c.execute(
            "SELECT COALESCE(SUM(end - start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        kernel_ns, n_kernels = int(row[0] or 0), int(row[1] or 0)

    # Sum of MEMCPY durations.
    if has_nvtx:
        f_st, f_en = nvtx_ranges['full_gen']
        p_st, p_en = nvtx_ranges['prefill_only']
        row_full = c.execute(
            f"SELECT COALESCE(SUM(end - start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            f"WHERE start >= {f_st} AND end <= {f_en}"
        ).fetchone()
        row_pref = c.execute(
            f"SELECT COALESCE(SUM(end - start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            f"WHERE start >= {p_st} AND end <= {p_en}"
        ).fetchone()
        full_v = int(row_full[0] or 0); pref_v = int(row_pref[0] or 0)
        full_n = int(row_full[1] or 0); pref_n = int(row_pref[1] or 0)
        memcpy_ns = int(max(full_v - pref_v, 0) * gen_tokens / max(gen_tokens - 1, 1))
        n_memcpy  = int(max(full_n - pref_n, 0) * gen_tokens / max(gen_tokens - 1, 1))
    else:
        row = c.execute(
            "SELECT COALESCE(SUM(end - start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchone()
        memcpy_ns, n_memcpy = int(row[0] or 0), int(row[1] or 0)

    # Helper: pull a CUDA runtime API stat, applying NVTX-decode-only
    # subtraction when ranges are present. Returns (sum_ns, count).
    def runtime_stat(like_clauses):
        # like_clauses is a list of LIKE patterns to OR together
        likes = ' OR '.join(f"s.value LIKE '{p}'" for p in like_clauses)
        if has_nvtx:
            f_st, f_en = nvtx_ranges['full_gen']
            p_st, p_en = nvtx_ranges['prefill_only']
            row_full = c.execute(f"""
SELECT COALESCE(SUM(r.end - r.start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE ({likes}) AND r.start >= {f_st} AND r.end <= {f_en}
""").fetchone()
            row_pref = c.execute(f"""
SELECT COALESCE(SUM(r.end - r.start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE ({likes}) AND r.start >= {p_st} AND r.end <= {p_en}
""").fetchone()
            sum_ns = int(max((row_full[0] or 0) - (row_pref[0] or 0), 0) * gen_tokens / max(gen_tokens - 1, 1))
            n     = int(max((row_full[1] or 0) - (row_pref[1] or 0), 0) * gen_tokens / max(gen_tokens - 1, 1))
            return sum_ns, n
        else:
            row = c.execute(f"""
SELECT COALESCE(SUM(r.end - r.start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE {likes}
""").fetchone()
            return int(row[0] or 0), int(row[1] or 0)

    launch_ns,       n_launch       = runtime_stat(["cudaLaunchKernel%"])
    sync_ns,         n_sync         = runtime_stat(["cudaStreamSynchronize%", "cudaDeviceSynchronize%"])
    memcpy_api_ns,   _              = runtime_stat(["cudaMemcpy%"])
    graph_launch_ns, n_graph_launch = runtime_stat(["cudaGraphLaunch%"])

    # All CUDA runtime API time
    if has_nvtx:
        f_st, f_en = nvtx_ranges['full_gen']
        p_st, p_en = nvtx_ranges['prefill_only']
        row_full = c.execute(
            f"SELECT COALESCE(SUM(end - start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            f"WHERE start >= {f_st} AND end <= {f_en}").fetchone()
        row_pref = c.execute(
            f"SELECT COALESCE(SUM(end - start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            f"WHERE start >= {p_st} AND end <= {p_en}").fetchone()
        runtime_total_ns = int(max((row_full[0] or 0) - (row_pref[0] or 0), 0) * gen_tokens / max(gen_tokens - 1, 1))
    else:
        row = c.execute(
            "SELECT COALESCE(SUM(end - start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ).fetchone()
        runtime_total_ns = int(row[0] or 0)

    other_api_ns = max(runtime_total_ns - launch_ns - sync_ns, 0)

    conn.close()

    # Per-token (ns → ms)
    ms = lambda x: x / 1_000_000 / gen_tokens
    out = {
        "framework": framework,
        "gen_tokens": gen_tokens,
        "decode_only_via_nvtx": bool(has_nvtx),
        "wall_ms_per_tok":     ms(wall_ns),
        "kernel_ms_per_tok":   ms(kernel_ns),
        "launch_ms_per_tok":   ms(launch_ns),
        "sync_ms_per_tok":     ms(sync_ns),
        "memcpy_ms_per_tok":   ms(memcpy_ns),
        "memcpy_api_ms_per_tok": ms(memcpy_api_ns),
        "graph_launch_ms_per_tok": ms(graph_launch_ns),
        "other_api_ms_per_tok": ms(other_api_ns),
        "tok_per_s":           1000.0 / ms(wall_ns) if wall_ns else 0,
        "n_kernels":           n_kernels,
        "n_kernels_per_tok":   n_kernels / gen_tokens,
        "n_launch_calls":      n_launch,
        "n_graph_launches":    n_graph_launch,
        "n_memcpy":            n_memcpy,
        "n_sync_calls":        n_sync,
        # Derived: cpu-side overhead = wall - kernel - memcpy_gpu  (GPU could overlap with CPU)
        # But for serial decode, residual = wall - kernel ≈ Python + sync wait time.
        "residual_ms_per_tok": ms(max(wall_ns - kernel_ns, 0)),
    }
    return out


def main():
    frameworks = ["trtllm", "llamacpp", "vllm", "pytorch"]
    rows = []
    for fw in frameworks:
        rep = TRACE_DIR / f"{fw}_decode.nsys-rep"
        if not rep.exists():
            print(f"[skip] {fw} (no trace)", file=sys.stderr)
            continue
        try:
            r = measure(rep, fw)
            rows.append(r)
            print(
                f"{fw:<10} wall={r['wall_ms_per_tok']:6.2f}ms "
                f"kernel={r['kernel_ms_per_tok']:6.2f}ms "
                f"launch={r['launch_ms_per_tok']:5.2f}ms "
                f"sync={r['sync_ms_per_tok']:5.2f}ms "
                f"residual={r['residual_ms_per_tok']:5.2f}ms "
                f"n_kern/tok={r['n_kernels_per_tok']:5.0f} "
                f"({r['tok_per_s']:.1f} tok/s)",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[err] {fw}: {e}", file=sys.stderr)
    out = ROOT / "breakdown.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}", file=sys.stderr)
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
