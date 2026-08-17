#!/bin/bash
# Unified sweep worker: prefill × decode × quantization × framework.
# Saves results incrementally to CSV (resume-safe: existing rows are skipped).
#
# Scope (set by run_orin_collection.sh):
#   SWEEP_SCOPE=full  (default)  all cells: 5 frameworks (incl. SGLang) ×
#                                {Llama-3.2-1B, Llama-3.1-8B, Mixtral} ×
#                                quants × pp/gen grid — the stage-1 master pass
#   SWEEP_SCOPE=1b               skips the 8B/Mixtral blocks — the per-rep
#                                stage-2 repeatability pass (~7 h each)
#
# History: this file unifies the original master sweep script (which had no
# SGLang runner) and the per-rep 1B script; the full scope now includes the
# SGLang cells that previously existed only in the rep passes.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_ROOT:-${PROFILE_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)/profile}}"
HF_DIR="${DATA_DIR}/models/hf_full"
MODELS_DIR="${DATA_DIR}/models"
ENGINES_DIR="${DATA_DIR}/models/trtllm_engines"
BENCH_DIR="${DATA_DIR}/benchmarks"
OUTPUT_DIR="${BENCH_DIR}/sweep_results"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/sweep_$(date +%Y%m%d_%H%M%S).csv}"

mkdir -p "${OUTPUT_DIR}"

# Set global paths early (bug fix: GGUF_DIR was previously set only in the
# llama.cpp section, making vllm 8B's `[ -f "${GGUF_DIR}/...gguf" ]` check
# silently false and skipping the 8B gguf_Q4_K_M branch).
GGUF_DIR="${MODELS_DIR}/gguf"

# ========================================================================
# Swap / cache preflight. We HARD-fail if there's enough swap in use to
# suggest active paging (> 3 GB). Lower amounts are usually pre-existing
# from other processes and the per-run swap_delta_kb watchdog will flag
# any benchmark that actually hits swap.
# ========================================================================
SWAP_USED_KB=$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)
SWAPPINESS=$(cat /proc/sys/vm/swappiness)
# Full master pass keeps the original strict 3 GB limit; the long rep
# series historically tolerated up to 5 GB of pre-existing swap.
SWAP_LIMIT_KB=3145728
[ "${SWEEP_SCOPE:-full}" = "1b" ] && SWAP_LIMIT_KB=5242880
if [ "$SWAP_USED_KB" -gt "$SWAP_LIMIT_KB" ]; then
    echo "ERROR: swap in use (${SWAP_USED_KB} KB > 3 GB). Run 'sudo ./preflight.sh'." >&2
    echo "  Current swap devices:" >&2
    swapon --show 2>&1 | sed 's/^/    /' >&2
    exit 1
elif [ "$SWAP_USED_KB" -gt 100 ]; then
    echo "WARN: ${SWAP_USED_KB} KB swap in use (pre-existing, not from benchmarks). Per-run watchdog will flag any deltas." >&2
fi
if [ "$SWAPPINESS" -gt 30 ]; then
    echo "WARNING: vm.swappiness=$SWAPPINESS (benchmarks want ≤10). Run 'sudo ./preflight.sh'." >&2
fi

# Record baseline swap for the watchdog (per-run delta)
SWAP_BASELINE_KB=$SWAP_USED_KB

# Sweep parameters
PREFILL_LENGTHS=(128 256 512 1024 2048 4096)
DECODE_LENGTHS=(128 256 512 1024 2048 4096)

# Common docker flags
DOCKER_COMMON="--rm --runtime nvidia -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v /proc/device-tree:/proc/device-tree:ro -e DEVICE_PROFILE=agx"

# CSV header — emits per-phase power rails, MFU/MBU, KV cache, device spec.
# Columns with *_measured vs *_roofline: measured = from tegrastats EMC util%;
# roofline = (weights + KV) / time / peak_bw. Legacy pp_mbu/dec_mbu = roofline.
# Only write header if file is new (allows appending to existing CSV via OUTPUT_CSV env override)
if [ ! -s "${OUTPUT_CSV}" ]; then
echo "timestamp,framework,model,quantization,prompt_tokens,gen_tokens,generated_tokens,\
ttft_ms,tpot_ms,prefill_tps,decode_tps,total_latency_ms,memory_mb,peak_memory_mb,\
idle_total_mw,idle_gpu_mw,idle_cpu_mw,idle_dram_mw,\
pp_total_mw,pp_gpu_mw,pp_cpu_mw,pp_soc_mw,pp_dram_mw,pp_gpu_util,pp_cpu_util,pp_emc_bw,pp_samples_warning,\
dec_total_mw,dec_gpu_mw,dec_cpu_mw,dec_soc_mw,dec_dram_mw,dec_gpu_util,dec_cpu_util,dec_emc_bw,dec_gpu_temp,dec_samples_warning,\
prefill_energy_mj,decode_energy_mj,prefill_gpu_energy_mj,decode_gpu_energy_mj,\
num_params,active_params,pp_tflops,dec_tflops,pp_attn_flops,dec_attn_flops,\
pp_mfu,dec_mfu,pp_mbu_measured,dec_mbu_measured,pp_mbu_roofline,dec_mbu_roofline,\
kv_cache_bytes_decode,peak_tflops_used,peak_tflops_fp16_dense,peak_bw_gb_s,device_name,\
swap_delta_kb,mem_available_kb_before,dec_mbu_roofline_total,active_weight_bytes" > "${OUTPUT_CSV}"
fi

echo "========================================"
echo "  Framework Sweep Benchmark"
echo "  Output: ${OUTPUT_CSV}"
echo "  Prefill: ${PREFILL_LENGTHS[*]}"
echo "  Decode:  ${DECODE_LENGTHS[*]}"
echo "========================================"

# Helper: extract field from JSON
jq_field() {
    echo "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print($2)" 2>/dev/null
}

# Helper: swap + memory snapshot for watchdog
_current_swap_kb() {
    awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo
}
_current_mem_avail_kb() {
    awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

# Helper: between-run cache / memory hygiene (called by run_*).
# Drops Python + docker ephemera, reports swap delta, and hard-fails if swap
# grew during the previous run (indicates model spilled to paging).
# Requires sudo -n drop_caches OR gracefully no-ops if not available.
_between_runs() {
    sync
    # Loud once-per-run warning instead of a silent no-op: on unified-memory
    # Jetson, an undropped page cache can starve GPU allocations mid-sweep
    # (observed: TRT engine build tactic OOM with 22 GB of page cache held).
    if ! sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
        if [ -z "${_DROP_CACHES_WARNED:-}" ]; then
            _DROP_CACHES_WARNED=1
            echo "  ⚠ WARNING: cannot drop page caches (no passwordless sudo)." >&2
            echo "    Free-memory pressure may distort results or fail engine builds." >&2
            echo "    Re-run under sudo -E, or grant NOPASSWD for: sh -c 'echo 3 > /proc/sys/vm/drop_caches'" >&2
        fi
    fi

    local cur_swap=$(_current_swap_kb)
    local delta=$((cur_swap - SWAP_BASELINE_KB))
    if [ "$delta" -gt 5120 ]; then   # > 5 MB grew — paging happened
        echo "  ⚠ swap grew ${delta} KB during last run (baseline ${SWAP_BASELINE_KB}, now ${cur_swap})" >&2
    fi
    SWAP_BASELINE_KB=$cur_swap
    LAST_MEM_AVAIL_KB=$(_current_mem_avail_kb)
}
LAST_MEM_AVAIL_KB=$(_current_mem_avail_kb)

# Helper: append result to CSV
append_csv() {
    local fw="$1" model="$2" quant="$3" pp="$4" gen="$5" json_str="$6"
    if [ -z "$json_str" ]; then
        echo "  SKIP: no result"
        return
    fi
    # Capture swap delta around this append (represents the run just completed)
    local swap_now=$(_current_swap_kb)
    local swap_delta_kb=$((swap_now - SWAP_BASELINE_KB))
    local mem_before_kb=$LAST_MEM_AVAIL_KB
    local row
    row=$(echo "$json_str" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ts = '$(date +%Y%m%d\ %H:%M:%S)'
fw, model, quant, pp, gen = '$fw', '$model', '$quant', '$pp', '$gen'
ip = d.get('idle_power', {})
ppp = d.get('prefill_power', {})
dp = d.get('decode_power', {})
def g(o, k, default=0):
    v = o.get(k, default)
    return v if v is not None else default
print(','.join(str(x) for x in [
    ts, fw, model, quant, pp, gen,
    d.get('generated_tokens', gen),  # actual tokens generated
    f\"{d.get('ttft_ms',0):.2f}\",
    f\"{d.get('tpot_ms',0):.2f}\",
    f\"{d.get('prefill_throughput_tps',0):.1f}\",
    f\"{d.get('decode_throughput_tps',0):.1f}\",
    f\"{d.get('total_latency_ms',0):.2f}\",
    f\"{d.get('memory_mb',0):.0f}\",
    f\"{d.get('peak_memory_mb',0):.0f}\",
    # idle rails
    g(ip, 'total_mw', g(ip, 'vdd_in_mw')),
    g(ip, 'gpu_mw', g(ip, 'vdd_cpu_gpu_cv_mw')),
    g(ip, 'cpu_mw', g(ip, 'vdd_cpu_gpu_cv_mw')),
    g(ip, 'dram_mw'),
    # prefill rails + util
    g(ppp, 'total_mw', g(ppp, 'vdd_in_mw')),
    g(ppp, 'gpu_mw', g(ppp, 'vdd_cpu_gpu_cv_mw')),
    g(ppp, 'cpu_mw', g(ppp, 'vdd_cpu_gpu_cv_mw')),
    g(ppp, 'soc_mw', g(ppp, 'vdd_soc_mw')),
    g(ppp, 'dram_mw'),
    g(ppp, 'gpu_util_pct'),
    g(ppp, 'cpu_util_pct'),
    g(ppp, 'emc_bw_gb_s'),
    int(bool(g(ppp, 'samples_warning'))),
    # decode rails + util + temp
    g(dp, 'total_mw', g(dp, 'vdd_in_mw')),
    g(dp, 'gpu_mw', g(dp, 'vdd_cpu_gpu_cv_mw')),
    g(dp, 'cpu_mw', g(dp, 'vdd_cpu_gpu_cv_mw')),
    g(dp, 'soc_mw', g(dp, 'vdd_soc_mw')),
    g(dp, 'dram_mw'),
    g(dp, 'gpu_util_pct'),
    g(dp, 'cpu_util_pct'),
    g(dp, 'emc_bw_gb_s'),
    g(dp, 'gpu_temp_c'),
    int(bool(g(dp, 'samples_warning'))),
    # energy
    f\"{d.get('prefill_energy_mj',0):.2f}\",
    f\"{d.get('decode_energy_mj',0):.2f}\",
    f\"{d.get('prefill_gpu_energy_mj',0):.2f}\",
    f\"{d.get('decode_gpu_energy_mj',0):.2f}\",
    # flops / mfu / mbu
    d.get('num_params', 0),
    d.get('active_params', d.get('num_params', 0)),
    d.get('prefill_tflops', 0),
    d.get('decode_tflops', 0),
    d.get('prefill_attn_flops', 0),
    d.get('decode_attn_flops', 0),
    d.get('pp_mfu', 0),
    d.get('dec_mfu', 0),
    d.get('pp_mbu_measured', d.get('pp_mbu', 0)),
    d.get('dec_mbu_measured', d.get('dec_mbu', 0)),
    d.get('pp_mbu_roofline', d.get('pp_mbu', 0)),
    d.get('dec_mbu_roofline', d.get('dec_mbu', 0)),
    d.get('kv_cache_bytes_decode', 0),
    d.get('peak_tflops_used', 0),
    d.get('peak_tflops_fp16_dense', 0),
    d.get('peak_bw_gb_s', 0),
    d.get('device_name', ''),
    $swap_delta_kb,
    $mem_before_kb,
    # MoE roofline columns (added 2026-04-24): all-weights pessimistic ceiling
    # and active-weight bytes used in dec_mbu_roofline. Both are 0 for dense.
    d.get('dec_mbu_roofline_total', 0),
    d.get('active_weight_bytes', 0),
]))
" 2>/dev/null)
    if [ -n "$row" ]; then
        echo "$row" >> "${OUTPUT_CSV}"
    fi
    # Between-run cleanup (drops page cache, updates swap baseline for next run)
    _between_runs
}

# Check if a run already exists in CSV (for resume support).
# Two calling forms:
#   run_exists "trtllm" "fp16" "128" "128"                       # 4 args, 1B path
#   run_exists "trtllm" "fp16" "128" "128" "Llama-3.1-8B"        # 5 args, 8B path
# Legacy callers also pass fw="trtllm,Llama-3.1-8B" — we detect that.
run_exists() {
    local fw="$1" quant="$2" pp="$3" gen="$4" model="${5:-}"
    if [[ "$fw" == *,* ]]; then
        # Legacy form: "fw,model" combined into first arg
        grep -q ",${fw},${quant},${pp},${gen}," "${OUTPUT_CSV}" 2>/dev/null
    elif [ -n "$model" ]; then
        grep -q ",${fw},${model},${quant},${pp},${gen}," "${OUTPUT_CSV}" 2>/dev/null
    else
        grep -q ",${fw},.*,${quant},${pp},${gen}," "${OUTPUT_CSV}" 2>/dev/null
    fi
}

# ============================================================
# TRT-LLM Sweep (Llama-3.2-1B: FP16, INT8, INT4)
# ============================================================
LLAMA_TOK="${HF_DIR}/llama-3.2-1b-instruct-untied"
TRTLLM_IMAGE="dustynv/tensorrt_llm:0.12-r36.4.0"

run_trtllm() {
    local engine_dir="$1" quant="$2" pp="$3" gen="$4"
    run_exists "trtllm" "$quant" "$pp" "$gen" && { echo "  [skip] trtllm ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  trtllm ${quant} pp=${pp} gen=${gen}..."
    local json
    json=$(docker run ${DOCKER_COMMON} \
        -v "${engine_dir}:/engine" \
        -v "${LLAMA_TOK}:/tokenizer" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_trtllm \
        "${TRTLLM_IMAGE}" \
        python3 /benchmarks/profiler_trtllm/bench_e2e.py \
            /engine /tokenizer "$pp" "$gen" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "trtllm" "Llama-3.2-1B" "$quant" "$pp" "$gen" "$json"
}

echo ""
echo "=== TRT-LLM Sweep (Llama-3.2-1B) ==="
for quant in fp16 int8 int4; do
    engine="${ENGINES_DIR}/llama-3.2-1b-instruct"
    [ "$quant" != "fp16" ] && engine="${ENGINES_DIR}/llama-3.2-1b-instruct-${quant}"
    [ ! -d "$engine" ] && { echo "  Engine not found: $engine"; continue; }
    for pp in "${PREFILL_LENGTHS[@]}"; do
        for gen in "${DECODE_LENGTHS[@]}"; do
            run_trtllm "$engine" "$quant" "$pp" "$gen"
        done
    done
done

# ---- 8B variant of run_trtllm (different tokenizer path) ----
LLAMA8B_TOK=$(find "${HF_DIR}/Llama-3.1-8B-Instruct" -maxdepth 0 -type d 2>/dev/null | head -1)
run_trtllm_8b() {
    local engine_dir="$1" quant="$2" pp="$3" gen="$4"
    run_exists "trtllm,Llama-3.1-8B" "$quant" "$pp" "$gen" && { echo "  [skip] trtllm 8B ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  trtllm 8B ${quant} pp=${pp} gen=${gen}..."
    local json
    json=$(docker run ${DOCKER_COMMON} \
        -v "${engine_dir}:/engine" \
        -v "${LLAMA8B_TOK}:/tokenizer" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_trtllm \
        "${TRTLLM_IMAGE}" \
        python3 /benchmarks/profiler_trtllm/bench_e2e.py \
            /engine /tokenizer "$pp" "$gen" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "trtllm" "Llama-3.1-8B" "$quant" "$pp" "$gen" "$json"
}

echo ""
if [ "${SWEEP_SCOPE:-full}" = "full" ]; then  # 8B/Mixtral — full scope only
echo "=== TRT-LLM Sweep (Llama-3.1-8B) ==="
# 8B engines built with MAX_SEQ=5120, so cap prefill+gen at ~5120.
for quant in fp16 int8 int4; do
    engine="${ENGINES_DIR}/llama-3.1-8b-instruct"
    [ "$quant" != "fp16" ] && engine="${ENGINES_DIR}/llama-3.1-8b-instruct-${quant}"
    [ ! -d "$engine" ] || [ ! -f "$engine/rank0.engine" ] && { echo "  Engine not found: $engine"; continue; }
    [ -z "$LLAMA8B_TOK" ] && { echo "  8B HF tokenizer not found"; continue; }
    for pp in "${PREFILL_LENGTHS[@]}"; do
        for gen in "${DECODE_LENGTHS[@]}"; do
            [ $((pp + gen)) -gt 5000 ] && { echo "  [skip] pp+gen=${pp}+${gen} > 5000 (engine max_seq=5120)"; continue; }
            run_trtllm_8b "$engine" "$quant" "$pp" "$gen"
        done
    done
done
fi  # end 8B/Mixtral (full scope only)

# ============================================================
# vLLM Sweep (Llama-3.2-1B: FP16, GGUF Q8_0, Q4_K_M, Q4_0)
# ============================================================
VLLM_IMAGE="dustynv/vllm:0.8.6-r36.4-cu128-24.04"
LLAMA_SNAP=$(find "${HF_DIR}/models--meta-llama--Llama-3.2-1B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)

run_vllm() {
    local model_arg="$1" mount_arg="$2" quant="$3" pp="$4" gen="$5"
    run_exists "vllm" "$quant" "$pp" "$gen" && { echo "  [skip] vllm ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  vllm ${quant} pp=${pp} gen=${gen}..."
    local json
    json=$(docker run ${DOCKER_COMMON} \
        ${mount_arg} \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_vllm \
        "${VLLM_IMAGE}" \
        python3 /benchmarks/profiler_vllm/bench_e2e.py \
            "${model_arg}" "$pp" "$gen" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "vllm" "Llama-3.2-1B" "$quant" "$pp" "$gen" "$json"
}

echo ""
echo "=== vLLM Sweep (Llama-3.2-1B) ==="
for pp in "${PREFILL_LENGTHS[@]}"; do
    for gen in "${DECODE_LENGTHS[@]}"; do
        # FP16
        run_vllm "/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(basename $LLAMA_SNAP)" \
            "-v ${HF_DIR}:/hf_models" "fp16" "$pp" "$gen"
        # GGUF formats
        for gguf in Q8_0 Q4_K_M Q4_0; do
            run_vllm "/models/gguf/Llama-3.2-1B-Instruct-${gguf}.gguf" \
                "-v ${MODELS_DIR}:/models" "gguf_${gguf}" "$pp" "$gen"
        done
    done
done

# ---- vLLM Llama-3.1-8B (reuses run_vllm but different model name in CSV) ----
run_vllm_8b() {
    local model_arg="$1" mount_arg="$2" quant="$3" pp="$4" gen="$5"
    run_exists "vllm,Llama-3.1-8B" "$quant" "$pp" "$gen" && { echo "  [skip] vllm 8B ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  vllm 8B ${quant} pp=${pp} gen=${gen}..."
    local json
    json=$(docker run ${DOCKER_COMMON} ${mount_arg} \
        -v "${BENCH_DIR}:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_vllm \
        "${VLLM_IMAGE}" \
        python3 /benchmarks/profiler_vllm/bench_e2e.py \
            "${model_arg}" "$pp" "$gen" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "vllm" "Llama-3.1-8B" "$quant" "$pp" "$gen" "$json"
}

echo ""
if [ "${SWEEP_SCOPE:-full}" = "full" ]; then  # 8B/Mixtral — full scope only
echo "=== vLLM Sweep (Llama-3.1-8B) ==="
if [ -n "$LLAMA8B_TOK" ]; then
    for pp in "${PREFILL_LENGTHS[@]}"; do
        for gen in "${DECODE_LENGTHS[@]}"; do
            # fp16 from HF dir (Llama-3.1-8B-Instruct, local-dir layout, no snapshots/)
            run_vllm_8b "/hf_models/Llama-3.1-8B-Instruct" \
                "-v ${HF_DIR}:/hf_models" "fp16" "$pp" "$gen"
            # Q4 GGUF
            if [ -f "${GGUF_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" ]; then
                run_vllm_8b "/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
                    "-v ${MODELS_DIR}:/models" "gguf_Q4_K_M" "$pp" "$gen"
            fi
        done
    done
else
    echo "  Llama-3.1-8B HF dir not found"
fi
fi  # end 8B/Mixtral (full scope only)

# ============================================================
# llama.cpp Sweep (Llama-3.2-1B: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q3_K_L)
# ============================================================
LLAMACPP_IMAGE="dustynv/llama_cpp:b5283-r36.4-cu128-24.04"
GGUF_DIR="${MODELS_DIR}/gguf"

run_llamacpp() {
    local gguf_file="$1" quant="$2" pp="$3" gen="$4" model_name="${5:-Llama-3.2-1B}"
    run_exists "llamacpp,${model_name}" "$quant" "$pp" "$gen" && { echo "  [skip] llamacpp ${model_name} ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  llamacpp ${model_name} ${quant} pp=${pp} gen=${gen}..."

    # Get context size from GGUF metadata
    local ctx
    ctx=$(docker run --rm -v "${GGUF_DIR}:/models" "${LLAMACPP_IMAGE}" \
        python3 -c "
import sys; sys.path.insert(0, '/data/benchmarks/profiler_llamacpp')
from read_gguf import read_metadata
info = read_metadata('/models/${gguf_file}')
print(info.get('context_length', 8192))
" 2>/dev/null) || ctx=8192
    # Ensure context is large enough for prefill + decode
    local min_ctx=$((pp + gen + 100))
    [ "$ctx" -lt "$min_ctx" ] && ctx="$min_ctx"

    # gpu_layers: 99 = all-on-GPU. For Mixtral-Q4 (~26 GB GGUF) on 32 GB AGX,
    # all-on-GPU OOMs at load time. Empirically gpu_layers=16 is the safe ceiling
    # (CUDA0 buffer 13.5 GB + KV + compute ~14.4 GB total — leaves headroom).
    # gpu_layers=24 / 28 also fail. Cuts decode speed (~3-4 tok/s vs ~11 at full GPU)
    # but lets the configuration measure something real.
    local gpu_layers=99
    local gguf_size_gb
    gguf_size_gb=$(stat -c '%s' "${GGUF_DIR}/${gguf_file}" 2>/dev/null | awk '{print int($1/1e9)}')
    if [ "${gguf_size_gb:-0}" -ge 24 ]; then
        gpu_layers=16
        echo "    [partial-offload] ${gguf_size_gb} GB GGUF → gpu_layers=${gpu_layers}"
    fi

    local json
    json=$(docker run ${DOCKER_COMMON} \
        -v "${DATA_DIR}:/data" \
        "${LLAMACPP_IMAGE}" \
        python3 /data/benchmarks/profiler_llamacpp/bench_e2e.py \
            "/data/models/gguf/${gguf_file}" "$ctx" "$pp" "$gen" "$gpu_layers" "$pp" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "llamacpp" "$model_name" "$quant" "$pp" "$gen" "$json"
}

echo ""

SGLANG_IMAGE="sglang-orin:0.4.6-sm87"

# ---- SGLang runner ----
run_sglang() {
    local model_arg="$1" mount_arg="$2" quant="$3" pp="$4" gen="$5"
    run_exists "sglang" "$quant" "$pp" "$gen" && { echo "  [skip] sglang ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  sglang ${quant} pp=${pp} gen=${gen}..."
    local json
    json=$(docker run ${DOCKER_COMMON} ${mount_arg} \
        -v "${BENCH_DIR}:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_sglang \
        "${SGLANG_IMAGE}" \
        python3 /benchmarks/profiler_sglang/bench_e2e.py \
            "${model_arg}" "$pp" "$gen" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "sglang" "Llama-3.2-1B" "$quant" "$pp" "$gen" "$json"
    _between_runs
}

echo ""
echo "=== SGLang Sweep (Llama-3.2-1B) ==="
for pp in "${PREFILL_LENGTHS[@]}"; do
    for gen in "${DECODE_LENGTHS[@]}"; do
        # SGLang: fp16 from HF (no GGUF support in profiler)
        run_sglang "/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(basename $LLAMA_SNAP)" \
            "-v ${HF_DIR}:/hf_models" "fp16" "$pp" "$gen"
    done
done


echo "=== llama.cpp Sweep ==="
declare -A LLAMA_GGUFS=(
    ["f16"]="Llama-3.2-1B-Instruct-f16.gguf"
    ["Q8_0"]="Llama-3.2-1B-Instruct-Q8_0.gguf"
    ["Q6_K"]="Llama-3.2-1B-Instruct-Q6_K.gguf"
    ["Q5_K_M"]="Llama-3.2-1B-Instruct-Q5_K_M.gguf"
    ["Q4_K_M"]="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ["Q4_0"]="Llama-3.2-1B-Instruct-Q4_0.gguf"
    ["Q3_K_L"]="Llama-3.2-1B-Instruct-Q3_K_L.gguf"
)

for quant in f16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_L; do
    gguf_file="${LLAMA_GGUFS[$quant]}"
    [ ! -f "${GGUF_DIR}/${gguf_file}" ] && { echo "  GGUF not found: ${gguf_file}"; continue; }
    for pp in "${PREFILL_LENGTHS[@]}"; do
        for gen in "${DECODE_LENGTHS[@]}"; do
            run_llamacpp "$gguf_file" "$quant" "$pp" "$gen" "Llama-3.2-1B"
        done
    done
done

# Llama-3.1-8B GGUF (Q4_K_M staged)
echo ""
if [ "${SWEEP_SCOPE:-full}" = "full" ]; then  # 8B/Mixtral — full scope only
echo "=== llama.cpp Sweep (Llama-3.1-8B) ==="
LLAMA8B_GGUF="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
if [ -f "${GGUF_DIR}/${LLAMA8B_GGUF}" ]; then
    for pp in "${PREFILL_LENGTHS[@]}"; do
        for gen in "${DECODE_LENGTHS[@]}"; do
            run_llamacpp "$LLAMA8B_GGUF" "Q4_K_M" "$pp" "$gen" "Llama-3.1-8B"
        done
    done
else
    echo "  8B GGUF not found: ${LLAMA8B_GGUF}"
fi

# Mixtral-8x7B MoE — two quants:
#   Q3_K_M ~21 GB (full GPU offload — bare label "Q3_K_M")
#   Q4_K_M ~26 GB (partial offload, gpu_layers=16 — labeled "Q4_K_M_gpu16"
#                 to distinguish from full-GPU runs in the dataset)
echo ""
fi  # end 8B/Mixtral (full scope only)
if [ "${SWEEP_SCOPE:-full}" = "full" ]; then  # 8B/Mixtral — full scope only
echo "=== llama.cpp Mixtral-8x7B Sweep ==="
MIXTRAL_GGUF_Q3="Mixtral-8x7B-Instruct-v0.1-Q3_K_M.gguf"
MIXTRAL_GGUF_Q4="Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf"
# Pair format: gguf_file:csv_label  — the csv_label encodes runtime config when
# it differs from the full-GPU baseline.
for mx_pair in "${MIXTRAL_GGUF_Q3}:Q3_K_M" "${MIXTRAL_GGUF_Q4}:Q4_K_M_gpu16"; do
    mx_file="${mx_pair%:*}"; mx_quant="${mx_pair#*:}"
    if [ -f "${GGUF_DIR}/${mx_file}" ]; then
        for pp in "${PREFILL_LENGTHS[@]}"; do
            for gen in "${DECODE_LENGTHS[@]}"; do
                run_llamacpp "$mx_file" "$mx_quant" "$pp" "$gen" "Mixtral-8x7B"
            done
        done
    else
        echo "  [skip] Mixtral GGUF not found: ${mx_file}"
    fi
done

# Cross-framework Mixtral on vLLM (loads same GGUF files via vllm GGUF support).
# Q3_K_M (21 GB) and Q4_K_M (26 GB) are the only quants that fit on AGX 32 GB.
run_vllm_mixtral() {
    local gguf_file="$1" quant="$2" pp="$3" gen="$4"
    run_exists "vllm,Mixtral-8x7B" "$quant" "$pp" "$gen" && { echo "  [skip] vllm Mixtral ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  vllm Mixtral ${quant} pp=${pp} gen=${gen}..."
    local json
    json=$(docker run ${DOCKER_COMMON} \
        -v "${MODELS_DIR}:/models" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_vllm \
        "${VLLM_IMAGE}" \
        python3 /benchmarks/profiler_vllm/bench_e2e.py \
            "/models/gguf/${gguf_file}" "$pp" "$gen" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "vllm" "Mixtral-8x7B" "$quant" "$pp" "$gen" "$json"
}

echo ""
fi  # end 8B/Mixtral (full scope only)
if [ "${SWEEP_SCOPE:-full}" = "full" ]; then  # 8B/Mixtral — full scope only
echo "=== vLLM Mixtral-8x7B Sweep ==="
for mx_pair in "${MIXTRAL_GGUF_Q3}:gguf_Q3_K_M" "${MIXTRAL_GGUF_Q4}:gguf_Q4_K_M"; do
    mx_file="${mx_pair%:*}"; mx_quant="${mx_pair#*:}"
    if [ -f "${GGUF_DIR}/${mx_file}" ]; then
        for pp in "${PREFILL_LENGTHS[@]}"; do
            for gen in "${DECODE_LENGTHS[@]}"; do
                run_vllm_mixtral "$mx_file" "$mx_quant" "$pp" "$gen"
            done
        done
    else
        echo "  [skip] Mixtral GGUF not found: ${mx_file}"
    fi
done
fi  # end 8B/Mixtral (full scope only)

# ============================================================
# PyTorch Sweep (Llama-3.2-1B: BF16, BF16+compile)
# ============================================================
PYTORCH_IMAGE="bitsandbytes-bench:r36.4.0"
LLAMA_SNAP=$(find "${HF_DIR}/models--meta-llama--Llama-3.2-1B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)

run_pytorch() {
    local quant="$1" pp="$2" gen="$3" compile_flag="$4"
    local fw_name="pytorch"
    [ -n "$compile_flag" ] && fw_name="pytorch_compile"
    run_exists "$fw_name" "$quant" "$pp" "$gen" && { echo "  [skip] ${fw_name} ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  ${fw_name} ${quant} pp=${pp} gen=${gen}..."

    local env_flags=""
    if [ -n "$compile_flag" ]; then
        env_flags="-e TORCH_COMPILE=1 -e TORCH_COMPILE_MODE=default -e TORCH_COMPILE_DYNAMIC=true"
    fi

    local json
    json=$(docker run ${DOCKER_COMMON} \
        -v "${HF_DIR}:/hf_models" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_pytorch \
        ${env_flags} \
        "${PYTORCH_IMAGE}" \
        python3 /benchmarks/profiler_pytorch/bench_e2e.py \
            "/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(basename $LLAMA_SNAP)" \
            "$pp" "$gen" "$quant" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "$fw_name" "Llama-3.2-1B" "$quant" "$pp" "$gen" "$json"
}

echo ""
echo "=== PyTorch Sweep (Llama-3.2-1B: bf16, 4bit, 8bit) ==="
for quant in bf16 4bit 8bit; do
    for pp in "${PREFILL_LENGTHS[@]}"; do
        for gen in "${DECODE_LENGTHS[@]}"; do
            run_pytorch "$quant" "$pp" "$gen" ""
        done
    done
done

# PyTorch + torch.compile sweep intentionally skipped for 32GB studies —
# torch.compile's graph-capture/recompilation overhead dominates at batch=1
# inference and its gains are orthogonal to our framework-comparison story.

# ---- PyTorch Llama-3.1-8B ----
run_pytorch_8b() {
    local quant="$1" pp="$2" gen="$3"
    run_exists "pytorch,Llama-3.1-8B" "$quant" "$pp" "$gen" && { echo "  [skip] pytorch 8B ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  pytorch 8B ${quant} pp=${pp} gen=${gen}..."
    local json
    json=$(docker run ${DOCKER_COMMON} \
        -v "${HF_DIR}:/hf_models" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_pytorch \
        "${PYTORCH_IMAGE}" \
        python3 /benchmarks/profiler_pytorch/bench_e2e.py \
            "/hf_models/Llama-3.1-8B-Instruct" \
            "$pp" "$gen" "$quant" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "pytorch" "Llama-3.1-8B" "$quant" "$pp" "$gen" "$json"
}

echo ""
if [ "${SWEEP_SCOPE:-full}" = "full" ]; then  # 8B/Mixtral — full scope only
echo "=== PyTorch Sweep (Llama-3.1-8B: 4bit, 8bit — no bf16, weights alone are 16 GB) ==="
if [ -n "$LLAMA8B_TOK" ]; then
    for quant in 4bit 8bit; do
        for pp in "${PREFILL_LENGTHS[@]}"; do
            for gen in "${DECODE_LENGTHS[@]}"; do
                run_pytorch_8b "$quant" "$pp" "$gen"
            done
        done
    done
else
    echo "  Llama-3.1-8B HF dir not found"
fi
fi  # end 8B/Mixtral (full scope only)

# ============================================================
# Summary
# ============================================================
TOTAL=$(tail -n +2 "${OUTPUT_CSV}" | wc -l)
echo ""
echo "========================================"
echo "  Sweep Complete!"
echo "  Total runs: ${TOTAL}"
echo "  Results: ${OUTPUT_CSV}"
echo "========================================"
echo ""
echo "Breakdown by framework:"
tail -n +2 "${OUTPUT_CSV}" | cut -d, -f2 | sort | uniq -c
echo ""
echo "Breakdown by quantization:"
tail -n +2 "${OUTPUT_CSV}" | cut -d, -f4 | sort | uniq -c