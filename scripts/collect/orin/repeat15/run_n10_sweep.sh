#!/bin/bash
# Repeatability — re-measure 15 headline cells with num_runs=10 to recover
# per-iteration σ. Captures the full bench_e2e.py JSON (which contains
# prefill_times_ms + decode_times_ms arrays) into per-cell files.
set -u

DATA_DIR=${DATA_ROOT:-/nvme/ispass/jetson-containers/data}
HF_DIR=$DATA_DIR/models/hf_full
GGUF_DIR=$DATA_DIR/models/gguf
ENGINES_DIR=$DATA_DIR/models/trtllm_engines
BENCH_DIR=$DATA_DIR/benchmarks
LLAMA_TOK=$HF_DIR/llama-3.2-1b-instruct-untied
LLAMA_SNAP=$(basename $(find $HF_DIR/models--meta-llama--Llama-3.2-1B-Instruct/snapshots -maxdepth 1 -mindepth 1 -type d | head -1))
MODEL_HF="/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$LLAMA_SNAP"

OUT_DIR=${PAPER_JETSON:-/nvme/ispass/paper_jetson}/repeat10
mkdir -p $OUT_DIR/results $OUT_DIR/logs

TRTLLM_IMG=dustynv/tensorrt_llm:0.12-r36.4.0
VLLM_IMG=dustynv/vllm:0.8.6-r36.4-cu128-24.04
LLAMACPP_IMG=dustynv/llama_cpp:b5283-r36.4-cu128-24.04
PYTORCH_IMG=bitsandbytes-bench:r36.4.0
SGLANG_IMG=sglang-orin:0.4.6-sm87

DOCKER_COMMON="--rm --runtime nvidia -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v /proc/device-tree:/proc/device-tree:ro -e DEVICE_PROFILE=agx"

N=10
MASTER_LOG=$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log

log()   { echo "[$(date +%T)] $*" | tee -a "$MASTER_LOG"; }
banner(){ echo "" | tee -a "$MASTER_LOG"; echo "============================================================" | tee -a "$MASTER_LOG"; echo "$*" | tee -a "$MASTER_LOG"; echo "============================================================" | tee -a "$MASTER_LOG"; }

# Extract the last JSON line from a raw log and save.
extract_json() {
    local raw=$1 out=$2
    grep '^{' "$raw" | tail -1 > "$out"
    local size=$(wc -c <"$out" 2>/dev/null || echo 0)
    log "  → JSON size $size bytes ($out)"
}

run_trtllm() {  # quant engine_subdir pp gen
    local quant=$1 engine=$2 pp=$3 gen=$4 tag="trtllm_${quant}_pp${pp}_gen${gen}"
    banner "$tag"
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1
    sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    docker run $DOCKER_COMMON \
        -v "${ENGINES_DIR}/${engine}:/engine" \
        -v "${LLAMA_TOK}:/tokenizer" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_trtllm \
        "$TRTLLM_IMG" \
        python3 /benchmarks/profiler_trtllm/bench_e2e.py \
            /engine /tokenizer "$pp" "$gen" $N \
        >$OUT_DIR/logs/${tag}.raw 2>&1
    extract_json $OUT_DIR/logs/${tag}.raw $OUT_DIR/results/${tag}.json
}

run_vllm() {  # quant model_path pp gen
    local quant=$1 mp=$2 pp=$3 gen=$4 tag="vllm_${quant}_pp${pp}_gen${gen}"
    banner "$tag"
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1
    sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    docker run $DOCKER_COMMON \
        -v "${HF_DIR}:/hf_models" \
        -v "${GGUF_DIR}:/models/gguf" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_vllm \
        "$VLLM_IMG" \
        python3 /benchmarks/profiler_vllm/bench_e2e.py \
            "$mp" "$pp" "$gen" $N \
        >$OUT_DIR/logs/${tag}.raw 2>&1
    extract_json $OUT_DIR/logs/${tag}.raw $OUT_DIR/results/${tag}.json
}

run_sglang() {
    local quant=$1 mp=$2 pp=$3 gen=$4 tag="sglang_${quant}_pp${pp}_gen${gen}"
    banner "$tag"
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1
    sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    docker run $DOCKER_COMMON \
        -v "${HF_DIR}:/hf_models" \
        -v "${GGUF_DIR}:/models/gguf" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_sglang \
        "$SGLANG_IMG" \
        python3 /benchmarks/profiler_sglang/bench_e2e.py \
            "$mp" "$pp" "$gen" $N \
        >$OUT_DIR/logs/${tag}.raw 2>&1
    extract_json $OUT_DIR/logs/${tag}.raw $OUT_DIR/results/${tag}.json
}

run_llamacpp() {  # quant gguf_file pp gen
    local quant=$1 gguf=$2 pp=$3 gen=$4 tag="llamacpp_${quant}_pp${pp}_gen${gen}"
    local ctx=$((pp + gen + 512))
    [ $ctx -lt 32768 ] && ctx=32768
    banner "$tag"
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1
    sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    docker run $DOCKER_COMMON \
        -v "${DATA_DIR}:/data" \
        "$LLAMACPP_IMG" \
        python3 /data/benchmarks/profiler_llamacpp/bench_e2e.py \
            "/data/models/gguf/$gguf" $ctx "$pp" "$gen" 99 "$pp" $N \
        >$OUT_DIR/logs/${tag}.raw 2>&1
    extract_json $OUT_DIR/logs/${tag}.raw $OUT_DIR/results/${tag}.json
}

run_pytorch() {  # quant pp gen
    local quant=$1 pp=$2 gen=$3 tag="pytorch_${quant}_pp${pp}_gen${gen}"
    banner "$tag"
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1
    sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    docker run $DOCKER_COMMON \
        -v "${HF_DIR}:/hf_models" \
        -v "${BENCH_DIR}:/benchmarks" \
        -e PYTHONPATH=/benchmarks/profiler_pytorch \
        "$PYTORCH_IMG" \
        python3 /benchmarks/profiler_pytorch/bench_e2e.py \
            "$MODEL_HF" \
            "$pp" "$gen" "$quant" $N \
        >$OUT_DIR/logs/${tag}.raw 2>&1
    extract_json $OUT_DIR/logs/${tag}.raw $OUT_DIR/results/${tag}.json
}

# =====================================================================
# 15 cells — each measured with num_runs=10
# =====================================================================
banner "REPEATABILITY n=$N SWEEP  ($(date))"

# --- pp=128 gen=128, fp16 (5 fws) ---
run_trtllm   fp16   llama-3.2-1b-instruct-fp16-128k    128 128
run_vllm     fp16   "$MODEL_HF"                        128 128
run_sglang   fp16   "$MODEL_HF"                        128 128
run_llamacpp f16    Llama-3.2-1B-Instruct-f16.gguf     128 128
run_pytorch  bf16                                      128 128

# --- pp=128 gen=128, 4-bit (5 fws) ---
run_trtllm   int4   llama-3.2-1b-instruct-int4-128k    128 128
run_vllm     Q4_K_M /models/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf 128 128
run_sglang   Q4_K_M /models/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf 128 128
run_llamacpp Q4_K_M Llama-3.2-1B-Instruct-Q4_K_M.gguf  128 128
run_pytorch  4bit                                      128 128

# --- pp=128 gen=4096, fp16 (5 fws) ---
run_trtllm   fp16   llama-3.2-1b-instruct-fp16-128k    128 4096
run_vllm     fp16   "$MODEL_HF"                        128 4096
run_sglang   fp16   "$MODEL_HF"                        128 4096
run_llamacpp f16    Llama-3.2-1B-Instruct-f16.gguf     128 4096
run_pytorch  bf16                                      128 4096

banner "DONE  ($(date))"
