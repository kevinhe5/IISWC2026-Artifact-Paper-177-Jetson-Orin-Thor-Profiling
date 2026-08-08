#!/bin/bash
# #1: quant ladder FULL 6x6 for llamacpp/vllm/trtedge (match Orin). Resumable, appends to *_FULL.csv.
set -uo pipefail
SD=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/scripts/benchmarks_thor_sweeps
OUT=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/data/sweep_results_agx_thor_128gb
LOG=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/logs/quant_grid.log
export VLLM_IMAGE="thor:r38.3.arm64-sbsa-cu130-24.04-vllm_0.12.0"
nohup bash -c 'while true; do pkill -9 -x rg 2>/dev/null; sleep 1; done' >/dev/null 2>&1 &
echo "######## QUANT GRID 6x6 $(date) ########" > "$LOG"
echo "===== trtedge quant 6x6 $(date) =====" | tee -a "$LOG"
bash "$SD/sweep_thor_trtedge_quantgrid.sh" >> "$LOG" 2>&1
echo "===== llamacpp quant 6x6 $(date) =====" | tee -a "$LOG"
OUTPUT_CSV="$OUT/sweep_thor_llamacpp_FULL.csv" bash "$SD/sweep_thor_llamacpp_quantgrid.sh" >> "$LOG" 2>&1
echo "===== vllm quant 6x6 $(date) =====" | tee -a "$LOG"
OUTPUT_CSV="$OUT/sweep_thor_vllm_FULL.csv" bash "$SD/sweep_thor_vllm_quantgrid.sh" >> "$LOG" 2>&1
echo "######## QUANT GRID ALL DONE $(date) ########" | tee -a "$LOG"
for fw in trtedge vllm llamacpp; do c="$OUT/sweep_thor_${fw}_FULL.csv"; echo "  $fw: $(($(wc -l <"$c")-1)) rows" ; done | tee -a "$LOG"
