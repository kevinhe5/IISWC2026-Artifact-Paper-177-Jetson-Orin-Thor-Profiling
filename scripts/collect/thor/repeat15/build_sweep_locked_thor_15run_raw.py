#!/usr/bin/env python3
"""Emit the Thor 15-run sweep as RAW rows (15 rows per cell) instead of per-cell means.
Reuses the exact cell->repeat_stats mapping from build_sweep_locked_thor_15run.py.
Throughput gguf cells (vLLM/SGLang @128/128 + 2048/2048) use the graphs-ON raw rows;
all other cells use the eager repeat_stats runs. Output: one 62-col CSV, 15 rows/cell,
which the plot generators average per (framework,quant,pp,gen)."""
import csv, json, os, sys
from pathlib import Path

# Source data roots on the collection machine — override via env to re-run elsewhere.
REPO = Path(__file__).resolve().parents[4]
PROFILE_ROOT = os.environ.get("PROFILE_ROOT", "/nvme/iiswc/Jetson_profile")
WORK = f"{PROFILE_ROOT}/work"   # intermediate/scratch under the profile root
MASTER = os.environ.get("THOR_MASTER_SWEEP",
    f"{PROFILE_ROOT}/data/benchmarks/thor/data/sweep_results_agx_thor_128gb/sweep_locked_thor_20260622.csv")
RSDIR  = f"{PROFILE_ROOT}/data/benchmarks/thor/data/repeat_stats"
RSDIR_TRT = f"{PROFILE_ROOT}/data/benchmarks/thor/data/repeat_stats_trtedge"
CUDAGRAPH = os.environ.get("THOR_CUDAGRAPH_ROWS", f"{WORK}/thor_cudagraph_15run_rows.csv")
# Output defaults into the repo (relative to this script), or pass an explicit path as argv[1].
OUT_CSV = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "data/chat/sweep_locked_thor.csv")

CSV_TO_JSON = {
 'ttft_ms':'ttft_ms','tpot_ms':'tpot_ms','prefill_tps':'prefill_throughput_tps','decode_tps':'decode_throughput_tps',
 'total_latency_ms':'total_latency_ms','memory_mb':'memory_mb','peak_memory_mb':'peak_memory_mb',
 'idle_total_mw':'idle_power.total_mw','idle_gpu_mw':'idle_power.gpu_mw','idle_cpu_mw':'idle_power.cpu_mw','idle_dram_mw':'idle_power.dram_mw',
 'pp_total_mw':'prefill_power.total_mw','pp_gpu_mw':'prefill_power.gpu_mw','pp_cpu_mw':'prefill_power.cpu_mw','pp_soc_mw':'prefill_power.soc_mw','pp_dram_mw':'prefill_power.dram_mw',
 'pp_gpu_util':'prefill_power.gpu_util_pct','pp_cpu_util':'prefill_power.cpu_util_pct','pp_emc_bw':'prefill_power.emc_bw_gb_s','pp_samples_warning':'prefill_power.samples_warning',
 'dec_total_mw':'decode_power.total_mw','dec_gpu_mw':'decode_power.gpu_mw','dec_cpu_mw':'decode_power.cpu_mw','dec_soc_mw':'decode_power.soc_mw','dec_dram_mw':'decode_power.dram_mw',
 'dec_gpu_util':'decode_power.gpu_util_pct','dec_cpu_util':'decode_power.cpu_util_pct','dec_emc_bw':'decode_power.emc_bw_gb_s',
 'dec_gpu_temp':'decode_power.gpu_temp_c','dec_samples_warning':'decode_power.samples_warning',
 'prefill_energy_mj':'prefill_energy_mj','decode_energy_mj':'decode_energy_mj','prefill_gpu_energy_mj':'prefill_gpu_energy_mj','decode_gpu_energy_mj':'decode_gpu_energy_mj',
 'num_params':'num_params','active_params':'active_params','pp_tflops':'prefill_tflops','dec_tflops':'decode_tflops',
 'pp_attn_flops':'prefill_attn_flops','dec_attn_flops':'decode_attn_flops','pp_mfu':'pp_mfu','dec_mfu':'dec_mfu',
 'pp_mbu_measured':'pp_mbu_measured','dec_mbu_measured':'dec_mbu_measured','pp_mbu_roofline':'pp_mbu_roofline','dec_mbu_roofline':'dec_mbu_roofline',
 'kv_cache_bytes_decode':'kv_cache_bytes_decode','peak_tflops_used':'peak_tflops_used','peak_tflops_fp16_dense':'peak_tflops_fp16_dense','peak_bw_gb_s':'peak_bw_gb_s',
}
BOOL_COLS = {'pp_samples_warning','dec_samples_warning'}


def jget(d, path):
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def map_repeat_name(fw, quant):
    if fw == 'vllm':   return ('vllm', quant)
    if fw == 'sglang': return ('sglang', quant)
    if fw == 'llamacpp':        return ('llama', 'fp16' if quant == 'f16' else quant)
    if fw == 'pytorch':         return ('pytorch_eager', 'fp16' if quant == 'bf16' else quant)
    if fw == 'pytorch_compile': return ('pytorch_compile', 'fp16' if quant == 'bf16' else quant)
    if fw == 'llamacpp_fa':     return ('llamacpp_fa', quant)
    if fw == 'llamacpp_fused':  return ('llamacpp_fused', quant)
    if fw == 'trtedge_llm':     return ('trtedge', quant)
    return None


def repeat_path(fw_tok, q_tok, pp, gen):
    d = RSDIR_TRT if fw_tok == 'trtedge' else RSDIR
    return os.path.join(d, f"{fw_tok}_{q_tok}_pp{pp}_gen{gen}.jsonl")


def load_runs(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# graphs-ON raw rows, keyed by cell -> list of full 62-col dict rows
cg = {}
if os.path.exists(CUDAGRAPH):
    for r in csv.DictReader(open(CUDAGRAPH)):
        cg.setdefault((r['framework'], r['quantization'], r['prompt_tokens'], r['gen_tokens']), []).append(r)

with open(MASTER) as fh:
    rd = csv.reader(fh); header = next(rd); master = list(rd)
col = {n: i for i, n in enumerate(header)}

out = []
n_cells_raw = n_cells_carry = 0
for row in master:
    fw, q = row[col['framework']], row[col['quantization']]
    pp, gen = row[col['prompt_tokens']], row[col['gen_tokens']]
    key = (fw, q, pp, gen)
    # 1) graphs-ON throughput cell -> ship its 15 raw rows verbatim (already 62-col)
    if key in cg:
        for cr in cg[key]:
            out.append([cr.get(h, row[col[h]]) for h in header])
        n_cells_raw += 1
        continue
    # 2) eager repeat_stats -> emit each run as a row
    m = map_repeat_name(fw, q)
    runs = load_runs(repeat_path(*m, pp, gen)) if (m and os.path.exists(repeat_path(*m, pp, gen))) else []
    if len(runs) >= 2:
        for rr in runs:
            nr = list(row)
            for c, jp in CSV_TO_JSON.items():
                v = jget(rr, jp)
                if v is None:
                    continue
                nr[col[c]] = str(int(bool(v))) if c in BOOL_COLS else str(v)
            out.append(nr)
        n_cells_raw += 1
    else:
        out.append(list(row))   # carry-through (no 15-run coverage) — 1 row
        n_cells_carry += 1

with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(header); w.writerows(out)
print(f"wrote {OUT_CSV}: {len(out)} rows ; cells with raw runs={n_cells_raw} ; carry-through(1-row)={n_cells_carry}")
