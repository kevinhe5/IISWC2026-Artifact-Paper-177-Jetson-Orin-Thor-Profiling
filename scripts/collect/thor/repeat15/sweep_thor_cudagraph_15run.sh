#!/usr/bin/env bash
# 15-run CUDA-graphs-ON collection for the THROUGHPUT figure cells only (vLLM + SGLang gguf points).
# Re-creates the graphs-ON latency/energy data lost in the Thor reset, at N=15 fresh-load runs/cell.
# 62-col harness with graphs-ON flags. Resumable: a cell with >=N rows
# in OUT_CSV is skipped, so re-running continues where it stopped. Detach with setsid to survive caps.
# Cells: cells_{vllm,sglang}_15run.txt  (quant,pp,gen). fp16 stays EAGER (not collected here).
set -uo pipefail
N=${N:-15}
REPO=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}
DATA_DIR="$REPO/data"
GGUF_DIR="$DATA_DIR/models/gguf"; HF_DIR="$DATA_DIR/models/hf_full"; BENCH_DIR="$DATA_DIR/benchmarks/orin"
OUTDIR="$DATA_DIR/benchmarks/thor/data/sweep_results_agx_thor_128gb"
EAGER="$OUTDIR/sweep_locked_thor_20260622.csv"
WORK=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work
BUILD="$WORK/build_row.py"
VLLM_IMAGE="thor:r38.3.arm64-sbsa-cu130-24.04-vllm_0.12.0"
SGLANG_IMAGE="thor:r38.3.arm64-sbsa-cu130-24.04-sglang_0.5.7"
SNAP=9213176726f574b556790deb65791e0c5aa438b6
CP=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include
FIPATCH='sed -i "/^            kv_data_type,\$/a\\            None,  # o_data_type" /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flashinfer.py 2>/dev/null; true'

OUT_CSV="${OUT_CSV:-$WORK/thor_cudagraph_15run_rows.csv}"
LOG="${LOG:-$WORK/cudagraph_15run.log}"

clean() {
  pkill -9 -x rg 2>/dev/null
  docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
  sudo -n /usr/bin/jetson_clocks >/dev/null 2>&1 || true
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  local GC; GC=$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq 2>/dev/null)
  [ "$GC" = 1575000000 ] || echo "  WARN gpu freq=$GC (expected 1575000000)" | tee -a "$LOG"
  sleep 1
}

DOCKER_COMMON="--rm --runtime nvidia \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v /proc/device-tree:/proc/device-tree:ro \
  -e DEVICE_PROFILE=agx -e JETSON_PLATFORM=agx_thor_128gb"

model_mount() {
  local quant="$1"
  if [ "$quant" = "fp16" ]; then
    MODELP="/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$SNAP"; MOUNT="-v $HF_DIR:/hf_models"
  else
    local q="${quant#gguf_}"
    MODELP="/models/gguf/Llama-3.2-1B-Instruct-$q.gguf"; MOUNT="-v $DATA_DIR/models:/models"
  fi
}

# how many rows already in OUT_CSV for (fw,quant,pp,gen)
have_rows() {
  [ -f "$OUT_CSV" ] || { echo 0; return; }
  python3 - "$OUT_CSV" "$1" "$2" "$3" "$4" <<'PY'
import csv,sys
f,fw,q,pp,gen=sys.argv[1:6]; n=0
try:
    for r in csv.DictReader(open(f)):
        if r['framework']==fw and r['quantization']==q and r['prompt_tokens']==pp and r['gen_tokens']==gen: n+=1
except FileNotFoundError: pass
print(n)
PY
}

emit_vllm() { # quant pp gen
  local quant="$1" pp="$2" gen="$3"; model_mount "$quant"; clean
  local json
  json=$(docker run ${DOCKER_COMMON} ${MOUNT} \
      -v "$BENCH_DIR:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_vllm \
      -e VLLM_ENFORCE_EAGER=0 -e VLLM_ATTENTION_BACKEND=FLASHINFER \
      -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e CPATH="$CP" \
      "$VLLM_IMAGE" bash -c "$FIPATCH; python3 /benchmarks/profiler_vllm/bench_e2e.py '$MODELP' $pp $gen 1" \
      2>>"$LOG" | grep '^{' | tail -1)
  [ -z "$json" ] && { echo "  FAIL vllm $quant $pp $gen" | tee -a "$LOG"; return 1; }
  echo "$json" | python3 "$BUILD" vllm "$quant" "$pp" "$gen" "$(date '+%Y%m%d %H:%M:%S')" >> "$OUT_CSV"
}

emit_sglang() { # quant pp gen
  local quant="$1" pp="$2" gen="$3"; model_mount "$quant"; clean
  local json
  json=$(docker run ${DOCKER_COMMON} ${MOUNT} \
      -v "$BENCH_DIR:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_sglang \
      -e SGLANG_DISABLE_CUDA_GRAPH=0 \
      -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CPATH=/usr/local/cuda/targets/sbsa-linux/include \
      "$SGLANG_IMAGE" \
      python3 /benchmarks/profiler_sglang/bench_e2e.py "$MODELP" "$pp" "$gen" 1 \
      2>>"$LOG" | grep '^{' | tail -1)
  [ -z "$json" ] && { echo "  FAIL sglang $quant $pp $gen" | tee -a "$LOG"; return 1; }
  echo "$json" | python3 "$BUILD" sglang "$quant" "$pp" "$gen" "$(date '+%Y%m%d %H:%M:%S')" >> "$OUT_CSV"
}

runN() { # fw quant pp gen
  local fw="$1" q="$2" pp="$3" gen="$4"
  local have; have=$(have_rows "$fw" "$q" "$pp" "$gen")
  if [ "$have" -ge "$N" ]; then echo "  [$fw $q $pp/$gen] SKIP ($have>=$N)" | tee -a "$LOG"; return; fi
  local i
  for ((i=have+1; i<=N; i++)); do
    echo "  [$fw $q $pp/$gen] run $i/$N $(date +%T)" | tee -a "$LOG"
    emit_${fw} "$q" "$pp" "$gen" || true
  done
}

[ -s "$OUT_CSV" ] || head -1 "$EAGER" > "$OUT_CSV"
echo "## CUDA-GRAPH 15-RUN START N=$N $(date)" | tee -a "$LOG"
FW="${1:-both}"
if [ "$FW" = "vllm" ] || [ "$FW" = "both" ]; then
  while IFS=, read -r q pp gen; do runN vllm "$q" "$pp" "$gen"; done < "$WORK/cells_vllm_15run.txt"
fi
if [ "$FW" = "sglang" ] || [ "$FW" = "both" ]; then
  while IFS=, read -r q pp gen; do runN sglang "$q" "$pp" "$gen"; done < "$WORK/cells_sglang_15run.txt"
fi
echo "## DONE $(date): $(($(wc -l < $OUT_CSV)-1)) rows total" | tee -a "$LOG"
