#!/bin/bash
# ============================================================================
# side_sweeps.sh — the three small side-grids that back paper figure series
# beyond the master sweep. Invoked by run_orin_collection.sh (stage "side");
# not intended to be run standalone by reviewers, but safe to.
#
#   compile  PyTorch + torch.compile, 8 cells        → side_compile.csv
#            (Fig 5/6 "PyTorch(compile)" Orin series → data/chat/pytorch_compile.csv)
#   fa       llama.cpp flash-attn ON, 16 cells       → side_fa.csv
#            (Fig 5 "llama.cpp (FA on)" Orin series  → data/chat/llamacpp_fa_orin.csv)
#   longctx  long-decode gen 8K..131K, 27 cells      → side_longctx.csv
#            (Fig 6/8 long-context extension         → data/chat/longctx_fp16_orin.csv)
#
# Cell grids replicate the shipped data files exactly. All runs are eager,
# num_runs=1 (3 warm-up + 1 measured), locked clocks — same protocol as the
# master sweep. Every cell is resumable (present rows are skipped).
#
# WALL-CLOCK WARNING: longctx's 131K-token decode cells run ~40-70 min EACH
# (a 131 072-token generation at ~20-30 ms/token). Full longctx ≈ 8-10 h.
# compile ≈ 25 min. fa ≈ 30 min.
#
# TRT-LLM longctx cells need an engine with max_seq_len ≥ 131 328 (prepare's
# default engine is 8192-seq). This script builds one on demand (cached at
# models/trtllm_engines/llama-3.2-1b-instruct-long, reusing prepare's fp16
# checkpoint) — ~1 min one-time.
#
# Env:  SIDE=compile|fa|longctx|all   (default all)
#       DATA_ROOT                      (profile root; set by the entry point)
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_ROOT:-${PROFILE_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)/profile}}"
HF_DIR="${DATA_DIR}/models/hf_full"
MODELS_DIR="${DATA_DIR}/models"
GGUF_DIR="${MODELS_DIR}/gguf"
ENGINES_DIR="${DATA_DIR}/models/trtllm_engines"
BENCH_DIR="${DATA_DIR}/benchmarks"
OUTPUT_DIR="${BENCH_DIR}/sweep_results"
SIDE="${SIDE:-all}"

mkdir -p "${OUTPUT_DIR}"

DOCKER_COMMON="--rm --runtime nvidia -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v /proc/device-tree:/proc/device-tree:ro -e DEVICE_PROFILE=agx"

TRTLLM_IMAGE="dustynv/tensorrt_llm:0.12-r36.4.0"
VLLM_IMAGE="dustynv/vllm:0.8.6-r36.4-cu128-24.04"
LLAMACPP_IMAGE="dustynv/llama_cpp:b5283-r36.4-cu128-24.04"
SGLANG_IMAGE="sglang-orin:0.4.6-sm87"
PYTORCH_IMAGE="bitsandbytes-bench:r36.4.0"

LLAMA_SNAP=$(find "${HF_DIR}/models--meta-llama--Llama-3.2-1B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
LLAMA_TOK="${HF_DIR}/llama-3.2-1b-instruct-untied"

# ---- swap watchdog + hygiene (same as sweep.sh) ---------------------------
SWAP_BASELINE_KB=$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)
_current_swap_kb()      { awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo; }
_current_mem_avail_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
LAST_MEM_AVAIL_KB=$(_current_mem_avail_kb)

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
    if [ "$delta" -gt 5120 ]; then
        echo "  ⚠ swap grew ${delta} KB during last run" >&2
    fi
    SWAP_BASELINE_KB=$cur_swap
    LAST_MEM_AVAIL_KB=$(_current_mem_avail_kb)
}

# ---- 62-col CSV header (identical to sweep.sh) ----------------------------
ensure_header() {
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
}

# ---- append_csv (verbatim from sweep.sh) ----------------------------------
append_csv() {
    local fw="$1" model="$2" quant="$3" pp="$4" gen="$5" json_str="$6"
    if [ -z "$json_str" ]; then
        echo "  SKIP: no result"
        return
    fi
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
    d.get('generated_tokens', gen),
    f\"{d.get('ttft_ms',0):.2f}\",
    f\"{d.get('tpot_ms',0):.2f}\",
    f\"{d.get('prefill_throughput_tps',0):.1f}\",
    f\"{d.get('decode_throughput_tps',0):.1f}\",
    f\"{d.get('total_latency_ms',0):.2f}\",
    f\"{d.get('memory_mb',0):.0f}\",
    f\"{d.get('peak_memory_mb',0):.0f}\",
    g(ip, 'total_mw', g(ip, 'vdd_in_mw')),
    g(ip, 'gpu_mw', g(ip, 'vdd_cpu_gpu_cv_mw')),
    g(ip, 'cpu_mw', g(ip, 'vdd_cpu_gpu_cv_mw')),
    g(ip, 'dram_mw'),
    g(ppp, 'total_mw', g(ppp, 'vdd_in_mw')),
    g(ppp, 'gpu_mw', g(ppp, 'vdd_cpu_gpu_cv_mw')),
    g(ppp, 'cpu_mw', g(ppp, 'vdd_cpu_gpu_cv_mw')),
    g(ppp, 'soc_mw', g(ppp, 'vdd_soc_mw')),
    g(ppp, 'dram_mw'),
    g(ppp, 'gpu_util_pct'),
    g(ppp, 'cpu_util_pct'),
    g(ppp, 'emc_bw_gb_s'),
    int(bool(g(ppp, 'samples_warning'))),
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
    f\"{d.get('prefill_energy_mj',0):.2f}\",
    f\"{d.get('decode_energy_mj',0):.2f}\",
    f\"{d.get('prefill_gpu_energy_mj',0):.2f}\",
    f\"{d.get('decode_gpu_energy_mj',0):.2f}\",
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
    d.get('dec_mbu_roofline_total', 0),
    d.get('active_weight_bytes', 0),
]))
" 2>/dev/null)
    if [ -n "$row" ]; then
        echo "$row" >> "${OUTPUT_CSV}"
    fi
    _between_runs
}

run_exists() {
    local fw="$1" quant="$2" pp="$3" gen="$4"
    grep -q ",${fw},.*,${quant},${pp},${gen}," "${OUTPUT_CSV}" 2>/dev/null
}

# ---- per-framework cell runners (patterns copied from sweep.sh) -----------

# run_pytorch_cell QUANT PP GEN FW_LABEL [COMPILE:0|1]
run_pytorch_cell() {
    local quant="$1" pp="$2" gen="$3" fw_name="$4" compile="${5:-0}"
    run_exists "$fw_name" "$quant" "$pp" "$gen" && { echo "  [skip] ${fw_name} ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  ${fw_name} ${quant} pp=${pp} gen=${gen}..."
    local env_flags=""
    [ "$compile" = "1" ] && env_flags="-e TORCH_COMPILE=1 -e TORCH_COMPILE_MODE=default -e TORCH_COMPILE_DYNAMIC=true"
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

# run_llamacpp_cell GGUF QUANT PP GEN FW_LABEL [FA:0|1]
run_llamacpp_cell() {
    local gguf_file="$1" quant="$2" pp="$3" gen="$4" fw_name="$5" fa="${6:-0}"
    run_exists "$fw_name" "$quant" "$pp" "$gen" && { echo "  [skip] ${fw_name} ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  ${fw_name} ${quant} pp=${pp} gen=${gen}..."
    local ctx=$((pp + gen + 100))
    [ "$ctx" -lt 8192 ] && ctx=8192
    local env_flags=""
    [ "$fa" = "1" ] && env_flags="-e FLASH_ATTN=1"
    local json
    json=$(docker run ${DOCKER_COMMON} \
        -v "${DATA_DIR}:/data" \
        ${env_flags} \
        "${LLAMACPP_IMAGE}" \
        python3 /data/benchmarks/profiler_llamacpp/bench_e2e.py \
            "/data/models/gguf/${gguf_file}" "$ctx" "$pp" "$gen" 99 "$pp" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "$fw_name" "Llama-3.2-1B" "$quant" "$pp" "$gen" "$json"
}

# run_vllm_cell QUANT PP GEN [NOCACHE:0|1]
run_vllm_cell() {
    local quant="$1" pp="$2" gen="$3" nocache="${4:-0}"
    run_exists "vllm" "$quant" "$pp" "$gen" && { echo "  [skip] vllm ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  vllm ${quant} pp=${pp} gen=${gen}..."
    local env_flags=""
    [ "$nocache" = "1" ] && env_flags="-e VLLM_DISABLE_PREFIX_CACHE=1"
    local json
    json=$(docker run ${DOCKER_COMMON} \
        -v "${HF_DIR}:/hf_models" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_vllm \
        ${env_flags} \
        "${VLLM_IMAGE}" \
        python3 /benchmarks/profiler_vllm/bench_e2e.py \
            "/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(basename $LLAMA_SNAP)" \
            "$pp" "$gen" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "vllm" "Llama-3.2-1B" "$quant" "$pp" "$gen" "$json"
}

# run_sglang_cell QUANT PP GEN [NOCACHE:0|1]
run_sglang_cell() {
    local quant="$1" pp="$2" gen="$3" nocache="${4:-0}"
    run_exists "sglang" "$quant" "$pp" "$gen" && { echo "  [skip] sglang ${quant} pp=${pp} gen=${gen}"; return; }
    echo "  sglang ${quant} pp=${pp} gen=${gen}..."
    local env_flags=""
    [ "$nocache" = "1" ] && env_flags="-e SGLANG_DISABLE_RADIX_CACHE=1"
    local json
    json=$(docker run ${DOCKER_COMMON} \
        -v "${HF_DIR}:/hf_models" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_sglang \
        ${env_flags} \
        "${SGLANG_IMAGE}" \
        python3 /benchmarks/profiler_sglang/bench_e2e.py \
            "/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(basename $LLAMA_SNAP)" \
            "$pp" "$gen" 1 \
        2>&1 | grep '^{' | tail -1)
    append_csv "sglang" "Llama-3.2-1B" "$quant" "$pp" "$gen" "$json"
}

# run_trtllm_cell ENGINE_DIR QUANT PP GEN
run_trtllm_cell() {
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

# ---- on-demand long-context TRT engine (max_seq 131 328) ------------------
LONG_ENGINE="${ENGINES_DIR}/llama-3.2-1b-instruct-long"
build_long_engine() {
    [ -f "${LONG_ENGINE}/rank0.engine" ] && { echo "  cached: ${LONG_ENGINE}"; return 0; }
    if [ ! -d "${ENGINES_DIR}/ckpt_fp16" ]; then
        echo "  ERROR: fp16 checkpoint missing (${ENGINES_DIR}/ckpt_fp16)." >&2
        echo "         Run prepare_orin.sh first — it builds the checkpoint." >&2
        return 1
    fi
    docker system prune -f >/dev/null 2>&1 || true
    # The engine build needs large contiguous unified-memory allocations;
    # held page cache starves it (observed: "Tactic Device request: 384MB
    # Available: 42MB" with 22 GB of cache). Drop caches, then warn if
    # available memory still looks too tight for the build to succeed.
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    avail_gb=$(( $(_current_mem_avail_kb) / 1024 / 1024 ))
    if [ "$avail_gb" -lt 12 ]; then
        echo "  ⚠ WARNING: only ${avail_gb} GB available RAM before TRT engine build." >&2
        echo "    The build may fail with 'Tactic Device request ... Available: ...MB'." >&2
        echo "    Free memory (close other jobs, drop caches as root) and re-run." >&2
    fi
    echo "  Building long-context TRT-LLM engine (max_seq 131328) → ${LONG_ENGINE}"
    mkdir -p "${LONG_ENGINE}"
    docker run --rm --runtime nvidia -i \
        -v "${ENGINES_DIR}:/engines" \
        "${TRTLLM_IMAGE}" bash -c "\
            trtllm-build --checkpoint_dir /engines/ckpt_fp16 --output_dir /engines/llama-3.2-1b-instruct-long \
                --gemm_plugin float16 --gpt_attention_plugin float16 \
                --max_input_len 4096 --max_seq_len 131328 --max_batch_size 1"
}

banner(){ echo; echo "======== $* ======== $(date -u +%FT%TZ)"; }

# ===========================================================================
# Grid A — PyTorch + torch.compile  (8 cells, replicates data/chat/pytorch_compile.csv)
# ===========================================================================
if [ "$SIDE" = "all" ] || [ "$SIDE" = "compile" ]; then
    banner "SIDE compile — PyTorch + torch.compile (8 cells)"
    OUTPUT_CSV="${OUTPUT_DIR}/side_compile.csv"; ensure_header
    for cell in 256:128 512:128 1024:128 2048:128 128:256 128:1024 128:2048 128:4096; do
        pp="${cell%:*}"; gen="${cell#*:}"
        run_pytorch_cell "bf16" "$pp" "$gen" "pytorch_compile" 1
    done
fi

# ===========================================================================
# Grid B — llama.cpp flash-attn ON  (16 cells, replicates data/chat/llamacpp_fa_orin.csv)
# ===========================================================================
if [ "$SIDE" = "all" ] || [ "$SIDE" = "fa" ]; then
    banner "SIDE fa — llama.cpp FLASH_ATTN=1 (16 cells)"
    OUTPUT_CSV="${OUTPUT_DIR}/side_fa.csv"; ensure_header
    F16_GGUF="Llama-3.2-1B-Instruct-f16.gguf"
    for pp in 128 256 512 768 1024 1536 2048 3072 4096 6144 8192 12288 16384; do
        run_llamacpp_cell "$F16_GGUF" "f16" "$pp" 32 "llamacpp_fa" 1
    done
    for gen in 4096 16384 32768; do
        run_llamacpp_cell "$F16_GGUF" "f16" 128 "$gen" "llamacpp_fa" 1
    done
fi

# ===========================================================================
# Grid C — long-context decode  (27 cells, replicates data/chat/longctx_fp16_orin.csv)
# ===========================================================================
if [ "$SIDE" = "all" ] || [ "$SIDE" = "longctx" ]; then
    banner "SIDE longctx — long-decode extension (27 cells, MANY HOURS)"
    OUTPUT_CSV="${OUTPUT_DIR}/side_longctx.csv"; ensure_header
    F16_GGUF="Llama-3.2-1B-Instruct-f16.gguf"

    # llama.cpp FA-off + FA-on
    for gen in 8192 16384 32768 65536 131072; do
        run_llamacpp_cell "$F16_GGUF" "f16" 128 "$gen" "llamacpp" 0
    done
    for gen in 16384 32768; do
        run_llamacpp_cell "$F16_GGUF" "f16_fa" 128 "$gen" "llamacpp_faon" 1
    done

    # PyTorch eager + compile
    for gen in 8192 16384 32768 65536; do
        run_pytorch_cell "bf16" 128 "$gen" "pytorch" 0
    done
    for gen in 8192 16384; do
        run_pytorch_cell "bf16" 128 "$gen" "pytorch_compile" 1
    done

    # SGLang (radix cache off) + vLLM (prefix cache off).
    # gen values match the shipped data: both engines cap the final cell just
    # under the model's 131 072 context (max_len headroom differs per engine).
    for gen in 8192 16384 32768 65536 130560; do
        run_sglang_cell "fp16_nocache" 128 "$gen" 1
    done
    for gen in 8192 16384 32768 65536 130944; do
        run_vllm_cell "fp16_nocache" 128 "$gen" 1
    done

    # TRT-LLM — needs the long-context engine (built on demand, cached)
    if build_long_engine; then
        for gen in 16384 32768 65536 131072; do
            run_trtllm_cell "${LONG_ENGINE}" "fp16" 128 "$gen"
        done
    else
        echo "  [skip] trtllm longctx cells — long engine unavailable" >&2
    fi
fi

banner "side_sweeps done"
