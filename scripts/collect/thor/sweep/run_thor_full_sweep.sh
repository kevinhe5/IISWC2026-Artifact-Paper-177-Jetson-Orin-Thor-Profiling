#!/bin/bash
# Master Thor FULL sweep — runs every framework's sweep_thor_*.sh over the full
# fp16 6x6 ISL×OSL grid + quant ladders, mirroring the Orin locked sweep.
# Resumable (each fw -> fixed CSV; cells skipped if already present). Locked clocks + rg watchdog.
set -uo pipefail
SD=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/scripts/benchmarks_thor_sweeps
OUT=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/data/sweep_results_agx_thor_128gb
LOG=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/logs/full_sweep.log
export VLLM_IMAGE="thor:r38.3.arm64-sbsa-cu130-24.04-vllm_0.12.0"
mkdir -p "$OUT"
# rg watchdog (DRAM contention guard) for the whole campaign
nohup bash -c 'while true; do pkill -9 -x rg 2>/dev/null; sleep 1; done' >/dev/null 2>&1 &
echo "WD_PID=$!"
echo "######## THOR FULL SWEEP $(date) ########" > "$LOG"
run_fw(){ local fw=$1 script=$2; local csv="$OUT/sweep_thor_${fw}_FULL.csv"
  echo "===== [$fw] $(date) -> $csv =====" | tee -a "$LOG"
  OUTPUT_CSV="$csv" bash "$SD/$script" >> "$LOG" 2>&1
  echo "===== [$fw] DONE $(date): $(grep -c , "$csv" 2>/dev/null) rows =====" | tee -a "$LOG"
}
# order: quick/medium first
run_fw trtedge   sweep_thor_trtedge.sh
run_fw vllm      sweep_thor_vllm.sh
run_fw sglang    sweep_thor_sglang.sh
run_fw llamacpp  sweep_thor_llamacpp.sh
run_fw pytorch   sweep_thor_pytorch.sh
run_fw llamacpp_fa    sweep_thor_llamacpp_fa.sh
run_fw llamacpp_fused sweep_thor_llamacpp_fused.sh
echo "######## THOR FULL SWEEP ALL DONE $(date) ########" | tee -a "$LOG"
for f in "$OUT"/sweep_thor_*_FULL.csv; do echo "  $(basename $f): $(($(wc -l <"$f")-1)) rows"; done | tee -a "$LOG"
