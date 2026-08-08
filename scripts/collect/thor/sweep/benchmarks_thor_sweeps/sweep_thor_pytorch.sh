#!/usr/bin/env bash
# Focused single-turn PyTorch sweep for Jetson AGX Thor — covers BOTH paper tiers:
#   - PyTorch eager   (Tier 3, framework="pytorch")          : TORCH_COMPILE unset
#   - PyTorch compile (Tier 2, framework="pytorch_compile")  : TORCH_COMPILE=1
# Uses the bitsandbytes image (transformers + bnb). bench CLI: <model> <pp> <gen> <quant> <runs>
# (quant is POSITIONAL here: bf16 / 4bit / 8bit), unlike the sglang/vllm benches.
#
# L-shape grid per mode (~13 cells): bf16 ISL sweep @gen128 + bf16 OSL sweep @pp128 + 4Kx4K corner.
# Quant ladder @128x128 (eager only): bnb 4bit (NF4) + 8bit (int8) -> Table 4 / PyTorch-NF4 host-tax.
#
# Usage:  bash sweep_thor_pytorch.sh        (dohbm is in docker group -> docker direct)
set -uo pipefail
REPO=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}
DATA_DIR="$REPO/data"; HF_DIR="$DATA_DIR/models/hf_full"; BENCH_DIR="$DATA_DIR/benchmarks/orin"
PT_IMAGE="${PT_IMAGE:-thor:r38.3.arm64-sbsa-cu130-24.04-bitsandbytes}"
LLAMA_SNAP=$(find "$HF_DIR/models--meta-llama--Llama-3.2-1B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
OUTDIR="$DATA_DIR/benchmarks/thor/data/sweep_results_agx_thor_128gb"; mkdir -p "$OUTDIR"
OUTPUT_CSV="${OUTPUT_CSV:-$OUTDIR/sweep_thor_pytorch_$(date +%Y%m%d_%H%M%S).csv}"

DOCKER_COMMON="--rm --runtime nvidia \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v /proc/device-tree:/proc/device-tree:ro \
  -e DEVICE_PROFILE=agx -e JETSON_PLATFORM=agx_thor_128gb \
  -e CPATH=/usr/local/cuda/targets/sbsa-linux/include"
# CPATH: torch.compile (Inductor) JIT-compiles triton kernels which need cuda.h; same Thor fix as SGLang.
# (Harmless for eager.) bitsandbytes image has transformers+bnb (bare pytorch_2.10 lacks transformers).

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

run_exists() { grep -q ",${1},Llama-3.2-1B,${2},${3},${4}," "$OUTPUT_CSV" 2>/dev/null; }  # fw,quant,pp,gen

append_csv() {  # fw quant pp gen json
    local fw="$1" quant="$2" pp="$3" gen="$4" json_str="$5"
    [ -z "$json_str" ] && { echo "  SKIP: no result ($fw $quant pp=$pp gen=$gen)"; return; }
    local row; row=$(echo "$json_str" | python3 -c "
import json,sys
d=json.load(sys.stdin); ts='$(date '+%Y%m%d %H:%M:%S')'; fw,quant,pp,gen='$fw','$quant','$pp','$gen'
ip=d.get('idle_power',{}); ppp=d.get('prefill_power',{}); dp=d.get('decode_power',{})
def g(o,k,dv=0):
    v=o.get(k,dv); return v if v is not None else dv
print(','.join(str(x) for x in [
 ts,fw,'Llama-3.2-1B',quant,pp,gen,d.get('generated_tokens',gen),
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
    sync
}

# Thor torch.compile needs: (1) the bundled triton ptxas-blackwell swapped for the system CUDA-13
# ptxas (only the latter knows sm_110a), and (2) >=2 runs so the bench drops run1 (the one-time
# compile, ~394ms decode) and reports steady-state. Eager: plain, 1 run.
PTXAS_SWAP='cp /usr/local/cuda/bin/ptxas /opt/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas-blackwell; '
run_cell() {  # fw mode quant pp gen
    local fw="$1" mode="$2" quant="$3" pp="$4" gen="$5"
    run_exists "$fw" "$quant" "$pp" "$gen" && { echo "  [skip] $fw $quant pp=$pp gen=$gen"; return; }
    echo "  [run] $fw $quant pp=$pp gen=$gen"
    local json
    if [ "$mode" = "compile" ]; then
        json=$(docker run ${DOCKER_COMMON} -e TORCH_COMPILE=1 -e TORCH_COMPILE_MODE=default -e TORCH_COMPILE_DYNAMIC=1 \
            -v "$HF_DIR:/hf_models" -v "$BENCH_DIR:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_pytorch \
            "$PT_IMAGE" \
            bash -c "${PTXAS_SWAP}python3 /benchmarks/profiler_pytorch/bench_e2e.py '$FP16_MODEL' '$pp' '$gen' '$quant' 3" \
            2>/dev/null | grep '^{' | tail -1)
    else
        json=$(docker run ${DOCKER_COMMON} \
            -v "$HF_DIR:/hf_models" -v "$BENCH_DIR:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_pytorch \
            "$PT_IMAGE" \
            python3 /benchmarks/profiler_pytorch/bench_e2e.py "$FP16_MODEL" "$pp" "$gen" "$quant" 1 \
            2>/dev/null | grep '^{' | tail -1)
    fi
    append_csv "$fw" "$quant" "$pp" "$gen" "$json"
}

LENGTHS=(128 256 512 1024 2048 4096)
FP16_MODEL="/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(basename "$LLAMA_SNAP")"

echo "===== PyTorch Thor sweep -> $OUTPUT_CSV ====="
for mode in eager compile; do
    fw="pytorch"; [ "$mode" = "compile" ] && fw="pytorch_compile"
    echo "--- $fw : bf16 ISL sweep @gen=128 ---"
    for pp in "${LENGTHS[@]}"; do run_cell "$fw" "$mode" bf16 "$pp" 128; done
    echo "--- $fw : bf16 OSL sweep @pp=128 ---"
    for gen in 256 512 1024 2048 4096; do run_cell "$fw" "$mode" bf16 128 "$gen"; done
    echo "--- $fw : bf16 4Kx4K corner ---"
    run_cell "$fw" "$mode" bf16 4096 4096
done
echo "--- pytorch eager quant ladder @128x128 (bnb) ---"
for q in 8bit 4bit; do run_cell pytorch eager "$q" 128 128; done
echo "===== DONE: $(grep -c , "$OUTPUT_CSV") rows in $OUTPUT_CSV ====="
