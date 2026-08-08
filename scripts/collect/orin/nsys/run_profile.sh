#!/bin/bash
# Run nsys profile on one of the four frameworks.
# Usage: ./run_profile.sh {trtllm|llamacpp|vllm|pytorch}
set -e

FW="$1"
if [ -z "$FW" ]; then
    echo "Usage: $0 {trtllm|llamacpp|vllm|pytorch}" >&2
    exit 1
fi

ROOT="${DATA_ROOT:-/nvme/ispass/jetson-containers/data}/benchmarks/nsys_profiles"
DATA_ROOT="${DATA_ROOT:-/nvme/ispass/jetson-containers/data}"
NSYS_ROOT="/opt/nvidia/nsight-systems/2024.5.4"
NSYS_BIN="${NSYS_ROOT}/target-linux-tegra-armv8/nsys"

TRACE_DIR="${ROOT}/traces"
LOG_DIR="${ROOT}/logs"
mkdir -p "${TRACE_DIR}" "${LOG_DIR}"

# Inside the container, the project root is bind-mounted at /work.
# Use container path for nsys output, then resolve back to host path for size check.
OUT_CONT="/work/traces/${FW}_decode"
OUT_HOST="${TRACE_DIR}/${FW}_decode"
LOG="${LOG_DIR}/${FW}.log"

# nsys profile flags — capture only the cudaProfilerStart/Stop window.
# --cuda-graph-trace=node breaks CUDA graphs (used by llama.cpp/trtllm decode)
# into individual node events so we can see real kernel durations.
NSYS_FLAGS=(
    profile
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --trace=cuda,osrt,nvtx
    --cuda-graph-trace=node
    --sample=none
    --force-overwrite=true
    --output="${OUT_CONT}"
)

case "$FW" in
    llamacpp)
        IMG="dustynv/llama_cpp:b5283-r36.4-cu128-24.04"
        SCRIPT="bench_llamacpp.py"
        ;;
    vllm)
        IMG="dustynv/vllm:0.8.6-r36.4-cu128-24.04"
        SCRIPT="bench_vllm.py"
        ;;
    pytorch)
        IMG="bitsandbytes-bench:r36.4.0"
        SCRIPT="bench_pytorch.py"
        ;;
    trtllm)
        IMG="dustynv/tensorrt_llm:0.12-r36.4.0"
        SCRIPT="bench_trtllm.py"
        ;;
    *)
        echo "unknown framework: $FW" >&2
        exit 1
        ;;
esac

echo "=== profiling $FW ===" | tee -a "${LOG}"
echo "image: ${IMG}" | tee -a "${LOG}"
echo "script: ${SCRIPT}" | tee -a "${LOG}"
echo "output: ${OUT_HOST}.nsys-rep" | tee -a "${LOG}"
echo "started: $(date)" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

# Run inside Docker. Mount nsight-systems read-only so nsys is available
# inside the container at the same path as on the host.
docker run --rm --runtime=nvidia \
    -v "${DATA_ROOT}:/data" \
    -v "${ROOT}:/work" \
    -v "${NSYS_ROOT}:${NSYS_ROOT}:ro" \
    -w /work \
    "${IMG}" \
    bash -c "${NSYS_BIN} ${NSYS_FLAGS[*]} python3 /work/scripts/${SCRIPT}" \
    2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "finished: $(date)" | tee -a "${LOG}"
echo "trace size: $(du -h "${OUT_HOST}.nsys-rep" | cut -f1)" | tee -a "${LOG}"
