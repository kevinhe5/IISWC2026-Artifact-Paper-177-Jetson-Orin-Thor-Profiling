#!/usr/bin/env bash
# TRT-Edge quant ladder FULL 6x6 (host-build runtime, power-instrumented). Appends to trtedge_FULL.csv (resumable).
set -uo pipefail
EDGE=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/TensorRT-Edge-LLM
PROF=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/orin/profiler_trtedge
HLIB=/usr/lib/aarch64-linux-gnu; CULIB=/usr/local/cuda/targets/sbsa-linux/lib
IMG=thor:r38.3.arm64-sbsa-cu130-24.04-transformers
C=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/data/sweep_results_agx_thor_128gb/sweep_thor_trtedge_FULL.csv
LENGTHS=(128 256 512 1024 2048 4096)
run_exists(){ grep -q ",trtedge_llm,Llama-3.2-1B,${1},${2},${3}," "$C" 2>/dev/null; }
cell(){ local q=$1 pp=$2 gen=$3
  run_exists "$q" "$pp" "$gen" && { echo "  [skip] $q $pp $gen"; return; }
  echo "  [run] trtedge $q pp=$pp gen=$gen"
  local j; j=$(docker run --rm --runtime nvidia -v "$EDGE:/repo" -v "$HLIB:/hl:ro" -v "$CULIB:/hcuda:ro" -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v "$PROF:/prof" -w /repo \
    -e LD_LIBRARY_PATH=/hl:/hcuda:/repo/build -e EDGELLM_PLUGIN_PATH=/repo/build/libNvInfer_edgellm_plugin.so -e TRT_BIN=/repo/build/examples/llm/llm_bench \
    "$IMG" python3 /prof/bench_e2e.py "/repo/workspace/quant/${q}_eng" "$pp" "$gen" 2>/dev/null)
  [ -z "$j" ] && { echo "    SKIP no json"; return; }
  echo "$j" | python3 -c "
import json,sys
d=json.load(sys.stdin); ts='$(date '+%Y%m%d %H:%M:%S')'; q,pp,gen='$q','$pp','$gen'
ip=d['idle_power'];ppp=d['prefill_power'];dp=d['decode_power']
def g(o,k,v=0):
 x=o.get(k,v);return x if x is not None else v
print(','.join(str(x) for x in [ts,'trtedge_llm','Llama-3.2-1B',q,pp,gen,d.get('generated_tokens',gen),
 f\"{d['ttft_ms']:.2f}\",f\"{d['tpot_ms']:.2f}\",f\"{d['prefill_throughput_tps']:.1f}\",f\"{d['decode_throughput_tps']:.1f}\",f\"{d['total_latency_ms']:.2f}\",0,0,
 g(ip,'total_mw'),g(ip,'gpu_mw'),g(ip,'cpu_mw'),g(ip,'dram_mw'),
 g(ppp,'total_mw'),g(ppp,'gpu_mw'),g(ppp,'cpu_mw'),g(ppp,'soc_mw'),g(ppp,'dram_mw'),g(ppp,'gpu_util_pct'),g(ppp,'cpu_util_pct'),g(ppp,'emc_bw_gb_s'),int(bool(g(ppp,'samples_warning'))),
 g(dp,'total_mw'),g(dp,'gpu_mw'),g(dp,'cpu_mw'),g(dp,'soc_mw'),g(dp,'dram_mw'),g(dp,'gpu_util_pct'),g(dp,'cpu_util_pct'),g(dp,'emc_bw_gb_s'),g(dp,'gpu_temp_c'),int(bool(g(dp,'samples_warning'))),
 f\"{d['prefill_energy_mj']:.2f}\",f\"{d['decode_energy_mj']:.2f}\",f\"{d['prefill_gpu_energy_mj']:.2f}\",f\"{d['decode_gpu_energy_mj']:.2f}\",
 d['num_params'],d['active_params'],d['prefill_tflops'],d['decode_tflops'],d['prefill_attn_flops'],d['decode_attn_flops'],
 d['pp_mfu'],d['dec_mfu'],d['pp_mbu_measured'],d['dec_mbu_measured'],d['pp_mbu_roofline'],d['dec_mbu_roofline'],
 d['kv_cache_bytes_decode'],d['peak_tflops_used'],d['peak_tflops_fp16_dense'],d['peak_bw_gb_s'],d['device_name'],0,0,
 d['dec_mbu_roofline_total'],d['active_weight_bytes']]))" >> "$C" && echo "    ok"
}
echo "===== TRT-Edge quant 6x6 -> $C ====="
for q in int8_sq int4_awq fp8 nvfp4; do for pp in "${LENGTHS[@]}"; do for gen in "${LENGTHS[@]}"; do cell "$q" "$pp" "$gen"; done; done; done
echo "===== trtedge quantgrid DONE: $(($(wc -l <"$C")-1)) rows ====="
