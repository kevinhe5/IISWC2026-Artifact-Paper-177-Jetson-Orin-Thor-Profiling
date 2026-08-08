#!/bin/bash
# Extend CV sweep from N=3 to N=14 (per-cell iterations 9 → 42, ≥ 30 target).
# Runs reps 4..14 sequentially using sweep_1B_only.sh.
set -u

BENCH_DIR=${DATA_ROOT:-/nvme/ispass/jetson-containers/data}/benchmarks
CUSTOM_SWEEP=${PAPER_JETSON:-/nvme/ispass/paper_jetson}/repeat10/sweep_1B_only.sh
OUT_DIR=${PAPER_JETSON:-/nvme/ispass/paper_jetson}/repeat10/full_cv
mkdir -p "$OUT_DIR"

MASTER="$OUT_DIR/master.log"
banner(){ echo "" | tee -a "$MASTER"; echo "===" | tee -a "$MASTER"; echo "[$(date +%F\ %T)] $*" | tee -a "$MASTER"; echo "===" | tee -a "$MASTER"; }

banner "EXTEND CV SWEEP  reps 4..14 (11 more)  target N=14 per cell"

preflight(){
    local running=$(docker ps -q 2>/dev/null | wc -l)
    if [ "$running" -gt 0 ]; then
        echo "  ERROR: $running docker container(s) still running" | tee -a "$MASTER"
        docker ps 2>&1 | tee -a "$MASTER"
        exit 1
    fi
    sudo -n /usr/bin/jetson_clocks 2>&1 | tee -a "$MASTER" >/dev/null
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    sleep 5
}

run_rep(){
    local rep=$1
    local csv="$OUT_DIR/rep${rep}.csv"
    local log="$OUT_DIR/rep${rep}.log"
    banner "REP ${rep} start  CSV=$csv"
    if [ -s "$csv" ]; then
        echo "  resuming rep${rep} with $(wc -l < $csv) existing rows" | tee -a "$MASTER"
    fi
    preflight
    cd "$BENCH_DIR"
    OUTPUT_CSV="$csv" bash "$CUSTOM_SWEEP" 2>&1 | tee -a "$log"
    banner "REP ${rep} DONE  rows=$(wc -l < $csv 2>/dev/null || echo 0)"
    docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
    pkill -f 'tegrastats|monitor.sh' 2>/dev/null || true
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    sleep 60
}

for r in 4 5 6 7 8 9 10 11 12 13 14; do
    run_rep $r
done

banner "EXTEND CV SWEEP DONE  $(date)"
