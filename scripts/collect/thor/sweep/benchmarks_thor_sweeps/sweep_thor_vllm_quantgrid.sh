#!/usr/bin/env bash
# Focused single-turn vLLM sweep for Jetson AGX Thor. Mirrors sweep_thor_llamacpp.sh:
# Thor vLLM image, JETSON_PLATFORM env, MAXN locked clocks, identical CSV schema, resumable.
#
# Grid:
#   - fp16 (HF safetensors): FULL 6x6 ISL x OSL grid   -> Figs 5/6/7/8
#   - GGUF Q8_0 / Q4_K_M / Q4_0 @ 128x128              -> Fig 9 / Fig 11 / Table 4 quant ladder
#
# Usage:  sg docker -c 'cd data/benchmarks && bash sweep_thor_vllm.sh'
set -uo pipefail
REPO=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}
DATA_DIR="$REPO/data"; GGUF_DIR="$DATA_DIR/models/gguf"; HF_DIR="$DATA_DIR/models/hf_full"; BENCH_DIR="$DATA_DIR/benchmarks/orin"
VLLM_IMAGE="${VLLM_IMAGE:-thor:r38.3.arm64-sbsa-cu130-24.04-vllm}"
LLAMA_SNAP=$(find "$HF_DIR/models--meta-llama--Llama-3.2-1B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
OUTDIR="$DATA_DIR/benchmarks/thor/data/sweep_results_agx_thor_128gb"; mkdir -p "$OUTDIR"
OUTPUT_CSV="${OUTPUT_CSV:-$OUTDIR/sweep_thor_vllm_$(date +%Y%m%d_%H%M%S).csv}"

DOCKER_COMMON="--rm --runtime nvidia \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v /proc/device-tree:/proc/device-tree:ro \
  -e DEVICE_PROFILE=agx -e JETSON_PLATFORM=agx_thor_128gb"

source "$BENCH_DIR/_lock_clocks.sh" && lock_clocks "$OUTDIR/clocks.log" || sudo -n /usr/bin/jetson_clocks

if [ ! -s "$OUTPUT_CSV" ]; then
echo "timestamp,framework,model,quantization,prompt_tokens,gen_tokens,generated_tokens,\
ttft_ms,tpot_ms,prefill_tps,decode_tps,total_latency_ms,memory_mb,peak_memory_mb,\
idle_total_mw,idle_gpu_mw,idle_cpu_mw,idle_dram_mw,\
pp_total_mw,pp_gpu_mw,pp_cpu_mw,pp_soc_mw,pp_dram_mw,pp_gpu_util,pp_cpu_util,pp_emc_bw,pp_samples_warning,\
dec_total_mw,dec_gpu_mw,dec_cpu_mw,dec_soc_mw,dec_dram_mw,dec_gpu_util,dec_cpu_util,dec_emc_bw,dec_gpu_temp,dec_samples_warning,\
prefill_energy_mj,decode_energy_mj,prefill_gpu_energy_mj,decode_gpu_energy_mj,\
num_params,active_params,pp_tflops,dec_tflops,pp_attn_flops,dec_attn_flops,\
pp_mfu,dec_mfu,pp_mbu_measured,dec_mbu_measured,pp_mbu_roofline,dec_mbu_roofline,\
kv_cache_bytes_decode,peak_tflops_used,peak_tflops_fp16_dense,peak_bw_gb_s,device_name,\
swap_delta_kb,mem_available_kb_before,dec_mbu_roofline_total,active_weight_bytes" > "$OUTPUT_CSV"
fi

run_exists() { grep -q ",vllm,Llama-3.2-1B,${1},${2},${3}," "$OUTPUT_CSV" 2>/dev/null; }  # quant,pp,gen

append_csv() {  # quant pp gen json
    local quant="$1" pp="$2" gen="$3" json_str="$4"
    [ -z "$json_str" ] && { echo "  SKIP: no result ($quant pp=$pp gen=$gen)"; return; }
    local row; row=$(echo "$json_str" | python3 -c "
import json,sys
d=json.load(sys.stdin); ts='$(date '+%Y%m%d %H:%M:%S')'; quant,pp,gen='$quant','$pp','$gen'
ip=d.get('idle_power',{}); ppp=d.get('prefill_power',{}); dp=d.get('decode_power',{})
def g(o,k,dv=0):
    v=o.get(k,dv); return v if v is not None else dv
print(','.join(str(x) for x in [
 ts,'vllm','Llama-3.2-1B',quant,pp,gen,d.get('generated_tokens',gen),
 f\"{d.get('ttft_ms',0):.2f}\",f\"{d.get('tpot_ms',0):.2f}\",
 f\"{d.get('prefill_throughput_tps',0):.1f}\",f\"{d.get('decode_throughput_tps',0):.1f}\",
 f\"{d.get('total_latency_ms',0):.2f}\",f\"{d.get('memory_mb',0):.0f}\",f\"{d.get('peak_memory_mb',0):.0f}\",
 g(ip,'total_mw'),g(ip,'gpu_mw'),g(ip,'cpu_mw'),g(ip,'dram_mw'),
 g(ppp,'total_mw'),g(ppp,'gpu_mw'),g(ppp,'cpu_mw'),g(ppp,'soc_mw'),g(ppp,'dram_mw'),
 g(ppp,'gpu_util_pct'),g(ppp,'cpu_util_pct'),g(ppp,'emc_bw_gb_s'),int(bool(g(ppp,'samples_warning'))),
 g(dp,'total_mw'),g(dp,'gpu_mw'),g(dp,'cpu_mw'),g(dp,'soc_mw'),g(dp,'dram_mw'),
 g(dp,'gpu_util_pct'),g(dp,'cpu_util_pct'),g(dp,'emc_bw_gb_s'),g(dp,'gpu_temp_c'),int(bool(g(dp,'samples_warning'))),
 f\"{d.get('prefill_energy_mj',0):.2f}\",f\"{d.get('decode_energy_mj',0):.2f}\",
 f\"{d.get('prefill_gpu_energy_mj',0):.2f}\",f\"{d.get('decode_gpu_energy_mj',0):.2f}\",
 d.get('num_params',0),d.get('active_params',d.get('num_params',0)),
 d.get('prefill_tflops',0),d.get('decode_tflops',0),d.get('prefill_attn_flops',0),d.get('decode_attn_flops',0),
 d.get('pp_mfu',0),d.get('dec_mfu',0),
 d.get('pp_mbu_measured',d.get('pp_mbu',0)),d.get('dec_mbu_measured',d.get('dec_mbu',0)),
 d.get('pp_mbu_roofline',d.get('pp_mbu',0)),d.get('dec_mbu_roofline',d.get('dec_mbu',0)),
 d.get('kv_cache_bytes_decode',0),d.get('peak_tflops_used',0),d.get('peak_tflops_fp16_dense',0),
 d.get('peak_bw_gb_s',0),d.get('device_name',''),0,0,
 d.get('dec_mbu_roofline_total',0),d.get('active_weight_bytes',0),
]))
" 2>/dev/null)
    [ -n "$row" ] && echo "$row" >> "$OUTPUT_CSV"
    sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
}

run_cell() {  # quant model_path mount_arg pp gen
    local quant="$1" model="$2" mount="$3" pp="$4" gen="$5"
    run_exists "$quant" "$pp" "$gen" && { echo "  [skip] $quant pp=$pp gen=$gen"; return; }
    echo "  [run] vllm $quant pp=$pp gen=$gen"
    local json; json=$(docker run ${DOCKER_COMMON} ${mount} \
        -v "$BENCH_DIR:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_vllm \
        "$VLLM_IMAGE" \
        python3 /benchmarks/profiler_vllm/bench_e2e.py "$model" "$pp" "$gen" 1 \
        2>/dev/null | grep '^{' | tail -1)
    append_csv "$quant" "$pp" "$gen" "$json"
}

LENGTHS=(128 256 512 1024 2048 4096)
FP16_MODEL="/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(basename "$LLAMA_SNAP")"
FP16_MOUNT="-v $HF_DIR:/hf_models"

echo "===== vLLM Thor sweep -> $OUTPUT_CSV ====="
echo "--- fp16 full 6x6 grid ---"
for pp in "${LENGTHS[@]}"; do for gen in "${LENGTHS[@]}"; do run_cell fp16 "$FP16_MODEL" "$FP16_MOUNT" "$pp" "$gen"; done; done
echo "--- GGUF quant ladder @128x128 ---"
for q in Q8_0 Q4_K_M Q4_0; do for pp in "${LENGTHS[@]}"; do for gen in "${LENGTHS[@]}"; do
    run_cell "gguf_$q" "/models/gguf/Llama-3.2-1B-Instruct-$q.gguf" "-v $DATA_DIR/models:/models" "$pp" "$gen"
done; done; done
echo "===== DONE: $(grep -c , "$OUTPUT_CSV") rows in $OUTPUT_CSV ====="
