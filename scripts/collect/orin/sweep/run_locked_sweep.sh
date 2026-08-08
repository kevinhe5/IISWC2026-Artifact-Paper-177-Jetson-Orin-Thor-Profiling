#!/bin/bash
# Re-run the main sweep with jetson_clocks LOCKED, to a new CSV
# (does not overwrite the existing default-config sweep_20260423_031540.csv).
#
# Background context: the existing sweep was captured with the schedutil
# governor (GPU bouncing 306↔1300 MHz). Locking clocks reveals 27-33%
# faster decode for GPU-bound frameworks (trtllm, llama.cpp) and 5-17%
# for Python-bound ones (vllm, pytorch) — see /architecture validation
# table for the methodology.
set -e

DATA=${DATA_ROOT:-/nvme/ispass/jetson-containers/data}
BENCH_DIR="${DATA}/benchmarks"
OUTPUT_DIR="${BENCH_DIR}/sweep_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="${OUTPUT_DIR}/sweep_locked_${TIMESTAMP}.csv"
LOG="${OUTPUT_DIR}/sweep_locked_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}"

# Sanity: clocks must be locked. Re-lock just to be sure.
echo "[$(date)] Locking Jetson clocks (sudo jetson_clocks)" | tee -a "${LOG}"
sudo -n /usr/bin/jetson_clocks 2>&1 | tee -a "${LOG}"
sudo -n /usr/bin/jetson_clocks --show 2>&1 | grep -E "GPU|EMC|NV Power" | tee -a "${LOG}"

# Sanity: nothing else running on the GPU
running=$(docker ps -q | wc -l)
if [ "$running" -gt 0 ]; then
    echo "ERROR: $running docker containers still running; refusing to start sweep" | tee -a "${LOG}"
    docker ps --format '{{.Image}} {{.Status}}' | tee -a "${LOG}"
    exit 1
fi

echo "[$(date)] Launching sweep.sh, output -> ${OUTPUT_CSV}" | tee -a "${LOG}"
echo "[$(date)] Tail this log to monitor: tail -f ${LOG}" | tee -a "${LOG}"

cd "${BENCH_DIR}"
OUTPUT_CSV="${OUTPUT_CSV}" ./sweep.sh 2>&1 | tee -a "${LOG}"

echo "[$(date)] sweep.sh finished" | tee -a "${LOG}"
echo "[$(date)] Output CSV: ${OUTPUT_CSV}" | tee -a "${LOG}"
