#!/bin/bash
# Reviewer #1 (extension): fill the FULL 6x6 pp/gen grid with N independent runs -> mean/std/CV,
# across all bench frameworks (fp16). Same runner/lock/watchdog as repeat_figs.sh; only the CELLS
# list is expanded to the full grid. Resume: a config with >=N rows is SKIP'd, so only missing
# pp/gen cells actually run (existing repeat_stats/*.jsonl are reused).
set -u
N=${N:-3}
DATA=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data
HF=$DATA/models/hf_full; GGUF=$DATA/models/gguf; BENCH=$DATA/benchmarks/orin
CP=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include
SNAP=$(find "$HF/models--meta-llama--Llama-3.2-1B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d|head -1)
HFM=/hf_models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(basename "$SNAP")
OUT=$DATA/benchmarks/thor/data/repeat_stats; mkdir -p "$OUT"
DC="--rm --runtime nvidia -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro -v /proc/device-tree:/proc/device-tree:ro -e DEVICE_PROFILE=agx -e JETSON_PLATFORM=agx_thor_128gb"
IMG_V=thor:r38.3.arm64-sbsa-cu130-24.04-vllm_0.12.0
IMG_S=thor:r38.3.arm64-sbsa-cu130-24.04-sglang_0.5.7
IMG_L=thor:r38.3.arm64-sbsa-cu130-24.04-llama_cpp_b5255
IMG_P=thor:r38.3.arm64-sbsa-cu130-24.04-bitsandbytes

( while :; do for p in $(pgrep -x rg); do kill -STOP "$p" 2>/dev/null; done; sleep 0.3; done ) & WD=$!
trap "kill $WD 2>/dev/null" EXIT
lock(){ sudo -n jetson_clocks >/dev/null 2>&1 || true
  G=$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq); E=$(cat /sys/class/devfreq/bwmgr/cur_freq)
  [ "$G" = 1575000000 ] && [ "$E" = 4266000000 ] || echo "  WARN clocks gpu=$G emc=$E"; }
clean(){ docker ps -aq|xargs -r docker rm -f >/dev/null 2>&1; lock; sudo -n tee /proc/sys/vm/drop_caches <<<3 >/dev/null 2>&1||true; sleep 2; }

# per-framework bench (pp gen) -> emits one JSON line. Matches each master sweep script exactly.
vllm(){ docker run $DC -v "$HF:/hf_models" -v "$BENCH:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_vllm \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CPATH="$CP" "$IMG_V" \
  python3 /benchmarks/profiler_vllm/bench_e2e.py "$HFM" "$1" "$2" 1; }
sglang(){ docker run $DC -v "$HF:/hf_models" -v "$BENCH:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_sglang \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CPATH=/usr/local/cuda/targets/sbsa-linux/include "$IMG_S" \
  python3 /benchmarks/profiler_sglang/bench_e2e.py "$HFM" "$1" "$2" 1; }
llama(){ local ctx=$(( $1 + $2 + 128 )); docker run $DC -v "$DATA:/data" "$IMG_L" \
  python3 /data/benchmarks/orin/profiler_llamacpp/bench_e2e.py "/data/models/gguf/Llama-3.2-1B-Instruct-f16.gguf" "$ctx" "$1" "$2" 99 "$1" 1; }
pytorch_eager(){ docker run $DC -v "$HF:/hf_models" -v "$BENCH:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_pytorch \
  -e CPATH=/usr/local/cuda/targets/sbsa-linux/include "$IMG_P" \
  python3 /benchmarks/profiler_pytorch/bench_e2e.py "$HFM" "$1" "$2" bf16 1; }
pytorch_compile(){ docker run $DC -v "$HF:/hf_models" -v "$BENCH:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_pytorch \
  -e TORCH_COMPILE=1 -e TORCH_COMPILE_MODE=default -e TORCH_COMPILE_DYNAMIC=1 -e CPATH=/usr/local/cuda/targets/sbsa-linux/include "$IMG_P" \
  bash -c "cp /usr/local/cuda/bin/ptxas /opt/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas-blackwell; python3 /benchmarks/profiler_pytorch/bench_e2e.py '$HFM' '$1' '$2' bf16 3"; }

runN(){ local label=$1 fn=$2 pp=$3 gen=$4; local f="$OUT/${label}.jsonl"
  local have=0; [ -f "$f" ] && have=$(grep -c . "$f" 2>/dev/null)
  if [ "$have" -ge "$N" ]; then echo "  [$label] SKIP ($have>=$N)"; return; fi
  for i in $(seq $((have+1)) $N); do clean; echo "  [$label] run $i/$N (have $have) $(date +%T)"
    local j; j=$("$fn" "$pp" "$gen" 2>/dev/null | grep '^{' | tail -1); [ -n "$j" ] && echo "$j" >>"$f" || echo "    (empty $i)"
  done; echo "  -> $(wc -l <"$f") runs"; }

GRID="128 256 512 1024 2048 4096"
FWS=${FWS:-"llama pytorch_eager sglang pytorch_compile vllm"}   # cheap->expensive; override to subset
echo "######## REPEAT GRID N=$N $(date) ########"
echo "## frameworks: $FWS ; grid: 6x6 = 36 cells each (missing-only)"
for fw in $FWS; do
  echo "==== $fw ===="
  for pp in $GRID; do for gen in $GRID; do
    runN "${fw}_fp16_pp${pp}_gen${gen}" "$fw" "$pp" "$gen"
  done; done
done
echo "######## DONE $(date) ########"
python3 ${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/scripts/agentic/agg_cv.py 2>&1 | tail -20
