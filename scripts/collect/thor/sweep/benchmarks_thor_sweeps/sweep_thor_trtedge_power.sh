#!/usr/bin/env bash
# TRT-Edge-LLM Thor sweep WITH power (62-col schema, matches Orin trtllm + other Thor frameworks).
# Uses profiler_trtedge/bench_e2e.py (tegrastats-windowed prefill/decode). fp16 full 6x6 + quant ladder @128x128.
set -uo pipefail
EDGE=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/TensorRT-Edge-LLM
PROF=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/orin/profiler_trtedge
BENCH=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/orin
IMG=trtedge_img2:latest
OUTDIR=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/data/sweep_results_agx_thor_128gb
CSV="${OUTPUT_CSV:-$OUTDIR/sweep_thor_trtedge_FULL.csv}"
mkdir -p "$OUTDIR"
source "$BENCH/_lock_clocks.sh" && lock_clocks "$OUTDIR/clocks.log" || sudo -n /usr/bin/jetson_clocks
DOCK="--rm --runtime nvidia -v $EDGE:/src -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v $PROF:/prof -w /src"
# 62-col header (same as other Thor sweeps)
if [ ! -s "$CSV" ] || head -1 "$CSV" | grep -q '^timestamp,framework,model,phase,'; then
echo "timestamp,framework,model,quantization,prompt_tokens,gen_tokens,generated_tokens,ttft_ms,tpot_ms,prefill_tps,decode_tps,total_latency_ms,memory_mb,peak_memory_mb,idle_total_mw,idle_gpu_mw,idle_cpu_mw,idle_dram_mw,pp_total_mw,pp_gpu_mw,pp_cpu_mw,pp_soc_mw,pp_dram_mw,pp_gpu_util,pp_cpu_util,pp_emc_bw,pp_samples_warning,dec_total_mw,dec_gpu_mw,dec_cpu_mw,dec_soc_mw,dec_dram_mw,dec_gpu_util,dec_cpu_util,dec_emc_bw,dec_gpu_temp,dec_samples_warning,prefill_energy_mj,decode_energy_mj,prefill_gpu_energy_mj,decode_gpu_energy_mj,num_params,active_params,pp_tflops,dec_tflops,pp_attn_flops,dec_attn_flops,pp_mfu,dec_mfu,pp_mbu_measured,dec_mbu_measured,pp_mbu_roofline,dec_mbu_roofline,kv_cache_bytes_decode,peak_tflops_used,peak_tflops_fp16_dense,peak_bw_gb_s,device_name,swap_delta_kb,mem_available_kb_before,dec_mbu_roofline_total,active_weight_bytes" > "$CSV"
fi
run_exists(){ grep -q ",trtedge_llm,Llama-3.2-1B,${1},${2},${3}," "$CSV" 2>/dev/null; }
append(){ local q="$1" pp="$2" gen="$3" json="$4"
  [ -z "$json" ] && { echo "  SKIP $q $pp $gen"; return; }
  echo "$json" | python3 -c "
import json,sys
d=json.load(sys.stdin); ts='$(date '+%Y%m%d %H:%M:%S')'; q,pp,gen='$q','$pp','$gen'
ip=d.get('idle_power',{});ppp=d.get('prefill_power',{});dp=d.get('decode_power',{})
def g(o,k,v=0):
    x=o.get(k,v); return x if x is not None else v
print(','.join(str(x) for x in [ts,'trtedge_llm','Llama-3.2-1B',q,pp,gen,d.get('generated_tokens',gen),
 f\"{d.get('ttft_ms',0):.2f}\",f\"{d.get('tpot_ms',0):.2f}\",f\"{d.get('prefill_throughput_tps',0):.1f}\",f\"{d.get('decode_throughput_tps',0):.1f}\",
 f\"{d.get('total_latency_ms',0):.2f}\",0,0,
 g(ip,'total_mw'),g(ip,'gpu_mw'),g(ip,'cpu_mw'),g(ip,'dram_mw'),
 g(ppp,'total_mw'),g(ppp,'gpu_mw'),g(ppp,'cpu_mw'),g(ppp,'soc_mw'),g(ppp,'dram_mw'),g(ppp,'gpu_util_pct'),g(ppp,'cpu_util_pct'),g(ppp,'emc_bw_gb_s'),int(bool(g(ppp,'samples_warning'))),
 g(dp,'total_mw'),g(dp,'gpu_mw'),g(dp,'cpu_mw'),g(dp,'soc_mw'),g(dp,'dram_mw'),g(dp,'gpu_util_pct'),g(dp,'cpu_util_pct'),g(dp,'emc_bw_gb_s'),g(dp,'gpu_temp_c'),int(bool(g(dp,'samples_warning'))),
 f\"{d.get('prefill_energy_mj',0):.2f}\",f\"{d.get('decode_energy_mj',0):.2f}\",f\"{d.get('prefill_gpu_energy_mj',0):.2f}\",f\"{d.get('decode_gpu_energy_mj',0):.2f}\",
 d.get('num_params',0),d.get('active_params',0),d.get('prefill_tflops',0),d.get('decode_tflops',0),d.get('prefill_attn_flops',0),d.get('decode_attn_flops',0),
 d.get('pp_mfu',0),d.get('dec_mfu',0),d.get('pp_mbu_measured',0),d.get('dec_mbu_measured',0),d.get('pp_mbu_roofline',0),d.get('dec_mbu_roofline',0),
 d.get('kv_cache_bytes_decode',0),d.get('peak_tflops_used',0),d.get('peak_tflops_fp16_dense',0),d.get('peak_bw_gb_s',0),d.get('device_name',''),0,0,
 d.get('dec_mbu_roofline_total',0),d.get('active_weight_bytes',0)]))" >> "$CSV" && echo "  -> $q $pp x $gen ok"
}
cell(){ local q="$1" eng="$2" pp="$3" gen="$4"
  run_exists "$q" "$pp" "$gen" && { echo "  [skip] $q $pp $gen"; return; }
  echo "  [run] trtedge $q pp=$pp gen=$gen"
  local j; j=$(docker run $DOCK $IMG python3 /prof/bench_e2e.py "$eng" "$pp" "$gen" 2>/dev/null)
  append "$q" "$pp" "$gen" "$j"
}
echo "===== TRT-Edge power sweep -> $CSV ====="
FP16=/src/workspace/Llama-3.2-1B/engines
echo "--- fp16 full 6x6 ---"
for pp in 128 256 512 1024 2048 4096; do for gen in 128 256 512 1024 2048 4096; do cell fp16 "$FP16" "$pp" "$gen"; done; done
echo "--- quant ladder @128x128 ---"
for q in int8_sq int4_awq fp8 nvfp4; do cell "$q" "/src/workspace/quant/${q}_eng" 128 128; done
echo "===== DONE: $(($(wc -l <"$CSV")-1)) rows ====="
