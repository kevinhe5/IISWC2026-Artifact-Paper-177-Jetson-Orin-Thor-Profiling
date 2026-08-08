#!/bin/bash
# Reviewer #1 (extension) — TRT-Edge QUANT ladder CV, mirroring sweep_thor_trtedge_quantgrid.sh.
# Prebuilt quant engines (workspace/quant/{int8_sq,int4_awq,fp8,nvfp4}_eng) run through the SAME
# bench_e2e.py the master used -> full rich JSON (ttft/tpot/power/energy), full 6x6 grid, N reps.
# (trtedge fp16 CV is the separable llm_bench axis in repeat_stats_trtedge/*prefill*|*decode* — done.)
# Resume: cell with >=N rows SKIP'd.
set -u
N=${N:-3}
EDGE=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/TensorRT-Edge-LLM
PROF=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/orin/profiler_trtedge
HLIB=/usr/lib/aarch64-linux-gnu; CULIB=/usr/local/cuda/targets/sbsa-linux/lib
IMG=thor:r38.3.arm64-sbsa-cu130-24.04-transformers
OUT=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/data/repeat_stats_trtedge; mkdir -p "$OUT"
LOG=${LOG_DIR:-/nvme/iiswc}/repeat_trtedge_quant_$(date +%Y%m%d_%H%M%S).log; ln -sf "$LOG" ${LOG_DIR:-/nvme/iiswc}/repeat_trtedge_quant_latest.log

( while :; do for p in $(pgrep -x rg); do kill -STOP "$p" 2>/dev/null; done; sleep 0.3; done ) & WD=$!
trap "kill $WD 2>/dev/null" EXIT
lock(){ sudo -n jetson_clocks >/dev/null 2>&1 || true
  G=$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq); E=$(cat /sys/class/devfreq/bwmgr/cur_freq)
  [ "$G" = 1575000000 ] && [ "$E" = 4266000000 ] || echo "  WARN clocks gpu=$G emc=$E"; }
clean(){ docker ps -aq|xargs -r docker rm -f >/dev/null 2>&1; lock; sudo -n tee /proc/sys/vm/drop_caches <<<3 >/dev/null 2>&1||true; sleep 2; }

trt_q(){ local q=$1 pp=$2 gen=$3
  docker run --rm --runtime nvidia -v "$EDGE:/repo" -v "$HLIB:/hl:ro" -v "$CULIB:/hcuda:ro" \
    -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v "$PROF:/prof" -w /repo \
    -e LD_LIBRARY_PATH=/hl:/hcuda:/repo/build -e EDGELLM_PLUGIN_PATH=/repo/build/libNvInfer_edgellm_plugin.so \
    -e TRT_BIN=/repo/build/examples/llm/llm_bench -e DEVICE_PROFILE=agx -e JETSON_PLATFORM=agx_thor_128gb \
    "$IMG" python3 /prof/bench_e2e.py "/repo/workspace/quant/${q}_eng" "$pp" "$gen"; }

runN(){ local label=$1 fn=$2; shift 2; local f="$OUT/${label}.jsonl"
  local have=0; [ -f "$f" ] && have=$(grep -c . "$f" 2>/dev/null)
  if [ "$have" -ge "$N" ]; then echo "  [$label] SKIP ($have>=$N)"; return; fi
  for i in $(seq $((have+1)) $N); do clean; echo "  [$label] run $i/$N (have $have) $(date +%T)"
    local j; j=$("$fn" "$@" 2>/dev/null | grep '^{' | tail -1); [ -n "$j" ] && echo "$j">>"$f" || echo "    (empty $i)"
  done; echo "  -> $(wc -l <"$f") runs"; }

GRID="128 256 512 1024 2048 4096"
echo "######## REPEAT TRTEDGE QUANT N=$N $(date) ########" | tee "$LOG"
for q in int8_sq int4_awq fp8 nvfp4; do echo "==== trtedge $q ====" | tee -a "$LOG"
  for pp in $GRID; do for gen in $GRID; do
    runN "trtedge_${q}_pp${pp}_gen${gen}" trt_q "$q" "$pp" "$gen" | tee -a "$LOG"
  done; done; done
echo "######## DONE $(date) ########" | tee -a "$LOG"
