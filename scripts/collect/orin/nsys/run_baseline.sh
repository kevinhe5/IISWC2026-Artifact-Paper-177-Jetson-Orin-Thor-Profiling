#!/bin/bash
# Run a framework's bench script WITHOUT nsys to get the baseline tok/s.
# Compare against the nsys-profiled run to isolate profiling overhead.
# Usage: ./run_baseline.sh {trtllm|llamacpp|vllm|pytorch}
set -e

FW="$1"
if [ -z "$FW" ]; then
    echo "Usage: $0 {trtllm|llamacpp|vllm|pytorch}" >&2
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-${PROFILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/profile}}"
ROOT="${DATA_ROOT}/benchmarks/nsys_profiles"
NSYS_ROOT="/opt/nvidia/nsight-systems/2024.5.4"

LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${FW}_baseline.log"

case "$FW" in
    llamacpp) IMG="dustynv/llama_cpp:b5283-r36.4-cu128-24.04"; SCRIPT="bench_llamacpp.py" ;;
    vllm)     IMG="dustynv/vllm:0.8.6-r36.4-cu128-24.04";       SCRIPT="bench_vllm.py" ;;
    pytorch)  IMG="bitsandbytes-bench:r36.4.0";                  SCRIPT="bench_pytorch.py" ;;
    trtllm)   IMG="dustynv/tensorrt_llm:0.12-r36.4.0";           SCRIPT="bench_trtllm.py" ;;
    *) echo "unknown framework: $FW" >&2; exit 1 ;;
esac

echo "=== baseline (no nsys) — $FW ===" | tee -a "${LOG}"
echo "image: ${IMG}" | tee -a "${LOG}"
echo "script: ${SCRIPT}" | tee -a "${LOG}"
echo "started: $(date)" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

docker run --rm --runtime=nvidia \
    -v "${DATA_ROOT}:/data" \
    -v "${ROOT}:/work" \
    -v "${NSYS_ROOT}:${NSYS_ROOT}:ro" \
    -w /work \
    "${IMG}" \
    python3 /work/scripts/${SCRIPT} \
    2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "finished: $(date)" | tee -a "${LOG}"
