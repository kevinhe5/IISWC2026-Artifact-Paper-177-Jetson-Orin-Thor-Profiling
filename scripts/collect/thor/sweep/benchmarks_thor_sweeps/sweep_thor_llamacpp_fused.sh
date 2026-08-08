#!/usr/bin/env bash
# llamacpp_fused (GGUF Q4 in CUDA graphs) sweep for Jetson AGX Thor (framework="llamacpp_fused").
# Same image/bench as sweep_thor_llamacpp.sh but FLASH_ATTN=1 (also enables the fused
# RoPE+KV-store op, which is gated on flash_attn=True). Directly tests the paper's Fig 5
# prefill af collapse: naive af=4.76e-5 -> FA af=2.38e-6 (O(I^2) traffic eliminated).
#
# L-shape grid (~14 cells): f16 ISL sweep @gen128 (prefill) + f16 OSL sweep @pp128 (decode)
# + 4Kx4K corner + quant ladder @128 (Q8_0/Q4_K_M/Q4_0). Resumable via OUTPUT_CSV.
set -uo pipefail
REPO=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}
DATA_DIR="$REPO/data"; GGUF_DIR="$DATA_DIR/models/gguf"; BENCH_DIR="$DATA_DIR/benchmarks/orin"
LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-thor:r38.3.arm64-sbsa-cu130-24.04-llama_cpp_b5255}"
OUTDIR="$DATA_DIR/benchmarks/thor/data/sweep_results_agx_thor_128gb"; mkdir -p "$OUTDIR"
OUTPUT_CSV="${OUTPUT_CSV:-$OUTDIR/sweep_thor_llamacpp_fused_$(date +%Y%m%d_%H%M%S).csv}"
FLASH_ATTN=1

SP=/opt/venv/lib/python3.12/site-packages
WRAP="$REPO/fused_compat/llama_cpp_fork"   # 0.3.9 py wrapper + fork sm_110 libs + missing-symbol patch
DOCKER_COMMON="--rm --runtime nvidia \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v /proc/device-tree:/proc/device-tree:ro \
  -e DEVICE_PROFILE=agx -e JETSON_PLATFORM=agx_thor_128gb -e FLASH_ATTN=${FLASH_ATTN} \
  -e LD_LIBRARY_PATH=${SP}/llama_cpp/lib -v ${WRAP}:${SP}/llama_cpp:ro"

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

run_exists() { grep -q ",llamacpp_fused,Llama-3.2-1B,${1},${2},${3}," "$OUTPUT_CSV" 2>/dev/null; }

append_csv() {
    local quant="$1" pp="$2" gen="$3" json_str="$4"
    [ -z "$json_str" ] && { echo "  SKIP: no result ($quant pp=$pp gen=$gen)"; return; }
    local row
    row=$(echo "$json_str" | python3 -c "
import json,sys
d=json.load(sys.stdin); ts='$(date '+%Y%m%d %H:%M:%S')'; quant,pp,gen='$quant','$pp','$gen'
ip=d.get('idle_power',{}); ppp=d.get('prefill_power',{}); dp=d.get('decode_power',{})
def g(o,k,dv=0):
    v=o.get(k,dv); return v if v is not None else dv
print(','.join(str(x) for x in [
 ts,'llamacpp_fused','Llama-3.2-1B',quant,pp,gen,d.get('generated_tokens',gen),
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

run_cell() {
    local quant="$1" gguf="$2" pp="$3" gen="$4"
    run_exists "$quant" "$pp" "$gen" && { echo "  [skip] $quant pp=$pp gen=$gen"; return; }
    local ctx=$((pp + gen + 128))
    echo "  [run] FA $quant pp=$pp gen=$gen ctx=$ctx"
    local json
    json=$(docker run ${DOCKER_COMMON} -v "$DATA_DIR:/data" "$LLAMACPP_IMAGE" \
        python3 /data/benchmarks/orin/profiler_llamacpp/bench_e2e.py \
            "/data/models/gguf/$gguf" "$ctx" "$pp" "$gen" 99 "$pp" 1 \
        2>/dev/null | grep '^{' | tail -1)
    append_csv "$quant" "$pp" "$gen" "$json"
}

declare -A G=(
 [f16]=Llama-3.2-1B-Instruct-f16.gguf [Q8_0]=Llama-3.2-1B-Instruct-Q8_0.gguf
 [Q4_K_M]=Llama-3.2-1B-Instruct-Q4_K_M.gguf [Q4_0]=Llama-3.2-1B-Instruct-Q4_0.gguf)
LENGTHS=(128 256 512 1024 2048 4096)

echo "===== llama.cpp+FA Thor sweep -> $OUTPUT_CSV ====="
echo "--- f16 ISL sweep @gen=128 (Fig5 prefill af collapse) ---"
for pp in "${LENGTHS[@]}"; do run_cell f16 "${G[f16]}" "$pp" 128; done
echo "--- f16 OSL sweep @pp=128 (Fig6 decode) ---"
for gen in 256 512 1024 2048 4096; do run_cell f16 "${G[f16]}" 128 "$gen"; done
echo "--- f16 4Kx4K corner ---"
run_cell f16 "${G[f16]}" 4096 4096
echo "--- quant ladder @128x128 ---"
for q in Q8_0 Q4_K_M Q4_0; do run_cell "$q" "${G[$q]}" 128 128; done
echo "===== DONE: $(grep -c , "$OUTPUT_CSV") rows in $OUTPUT_CSV ====="
