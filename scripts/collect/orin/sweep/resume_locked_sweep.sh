#!/bin/bash
# Resume the paused locked-clock sweep.
# sweep.sh has built-in resume: if a (framework, model, quant, pp, gen) row
# already exists in OUTPUT_CSV, it skips that cell. So we just point at the
# same CSV file and it continues where it left off.
#
# Existing partial CSV: sweep_locked_20260428_020532.csv (195 cells done)
# Tail log: tail -f /tmp/locked_sweep_resume.log
set -e

DATA=${DATA_ROOT:-/nvme/ispass/jetson-containers/data}
BENCH_DIR="${DATA}/benchmarks"
OUTPUT_DIR="${BENCH_DIR}/sweep_results"

# Use the SAME CSV path as the original run — sweep.sh skips already-present cells.
OUTPUT_CSV="${OUTPUT_DIR}/sweep_locked_20260428_020532.csv"
LOG="${OUTPUT_DIR}/sweep_locked_20260428_020532.log"

if [ ! -f "${OUTPUT_CSV}" ]; then
    echo "ERROR: expected partial CSV not found: ${OUTPUT_CSV}" >&2
    exit 1
fi

echo "[$(date)] resuming locked-clock sweep" | tee -a "${LOG}"
echo "[$(date)] partial CSV has $(wc -l < ${OUTPUT_CSV}) rows" | tee -a "${LOG}"
echo "[$(date)] last completed cell: $(tail -1 ${OUTPUT_CSV} | cut -d, -f2-7)" | tee -a "${LOG}"

# Re-lock clocks (in case anything reset them)
echo "[$(date)] re-locking jetson_clocks" | tee -a "${LOG}"
sudo -n /usr/bin/jetson_clocks 2>&1 | tee -a "${LOG}"
sudo -n /usr/bin/jetson_clocks --show 2>&1 | grep -E "GPU MinFreq|EMC|NV Power" | tee -a "${LOG}"

# Sanity: nothing else on GPU
running=$(docker ps -q | wc -l)
if [ "$running" -gt 0 ]; then
    echo "ERROR: $running docker container(s) running, refusing to resume" | tee -a "${LOG}"
    docker ps --format '{{.Image}} {{.Status}}' | tee -a "${LOG}"
    exit 1
fi

cd "${BENCH_DIR}"
echo "[$(date)] ./sweep.sh starting (resume mode — already-done cells will be SKIP'd)" | tee -a "${LOG}"
OUTPUT_CSV="${OUTPUT_CSV}" ./sweep.sh 2>&1 | tee -a "${LOG}"

echo "[$(date)] sweep.sh resume finished" | tee -a "${LOG}"
echo "[$(date)] CSV has $(wc -l < ${OUTPUT_CSV}) rows" | tee -a "${LOG}"
