#!/bin/bash
# Parameterized decode-only nsys capture for Thor (Fig6/7/8 artifact gaps).
# Usage: nsys_capture.sh <fw> <gen> <tag> [warmup]
#   fw   = llamacpp | pytorch | sglang | vllm
#   gen  = decode tokens (128 short, 65536 long)
#   tag  = output basename (trace -> $OUT/<tag>.nsys-rep)
# Reuses the exact proven docker invocations from fig10_long_clean.sh.
# Writes trace to $OUT (staging). One capture at a time. Clean-run discipline.
set -u
FW="$1"; G="$2"; TAG="$3"; NW="${4:-3}"
HF="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/models/hf_full"
GGUF="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/models/gguf"
SCR="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/orin/nsys_profiles"   # holds scripts/nsys_*.py (mounted ro)
OUT="${NSYS_TRACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/data/nsys/traces}"
HFM="/hf/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
CP="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include"
NSROOT="/opt/nvidia/nsight-systems/2025.6.3"; NSYS="$NSROOT/target-linux-sbsa-armv8/nsys"; NSMNT="-v $NSROOT:$NSROOT:ro"
NFC="--capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda --cuda-graph-trace=node --sample=none --force-overwrite=true"
NFF="--trace=cuda --trace-fork-before-exec=true --sample=none --force-overwrite=true"
LOG="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work/logs/cap_${TAG}.log"

clean(){ pkill -9 -x rg 2>/dev/null; docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
  sudo nvpmodel -m 0 >/dev/null 2>&1 || true; sudo jetson_clocks >/dev/null 2>&1 || true
  sudo tee /proc/sys/vm/drop_caches <<<3 >/dev/null 2>&1 || true
  GC=$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq)
  [ "$GC" = 1575000000 ] || { echo "CLOCK UNLOCKED $GC ABORT" | tee -a "$LOG"; exit 1; }; sleep 2; }

echo "######## CAPTURE $FW gen=$G tag=$TAG $(date) ########" > "$LOG"
clean

case "$FW" in
  llamacpp)
    timeout 7000 docker run --rm --runtime=nvidia \
      -e BENCH_MODEL="/gguf/Llama-3.2-1B-Instruct-f16.gguf" -e BENCH_PROMPT_LEN=128 \
      -e BENCH_GEN_TOKENS=$G -e BENCH_NWARMUP=$NW -e BENCH_FLASH=1 \
      -v "$GGUF:/gguf:ro" -v "$SCR:/work:ro" -v "$OUT:/out" $NSMNT -w /work \
      thor:r38.3.arm64-sbsa-cu130-24.04-llama_cpp_b5255 \
      bash -c "$NSYS profile $NFC -o /out/$TAG python3 /work/scripts/nsys_llamacpp.py" >>"$LOG" 2>&1 ;;
  pytorch)
    timeout 14400 docker run --rm --runtime=nvidia \
      -e BENCH_MODEL="$HFM" -e PYTORCH_ATTN_IMPL=eager -e BENCH_DTYPE=bf16 -e BENCH_NWARMUP=$NW \
      -e CPATH="$CP" -e BENCH_PROMPT_LEN=128 -e BENCH_GEN_TOKENS=$G \
      -v "$HF:/hf:ro" -v "$SCR:/work:ro" -v "$OUT:/out" $NSMNT -w /work \
      thor:r38.3.arm64-sbsa-cu130-24.04-transformers \
      bash -c "$NSYS profile $NFC -o /out/$TAG python3 /work/scripts/nsys_pytorch.py" >>"$LOG" 2>&1 ;;
  sglang)
    timeout 7000 docker run --rm --runtime=nvidia \
      -e BENCH_MODEL="$HFM" -e BENCH_PROMPT_LEN=128 -e BENCH_GEN_TOKENS=$G \
      -e BENCH_ATTN_BACKEND=flashinfer -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CPATH="$CP" \
      -v "$HF:/hf:ro" -v "$SCR:/work:ro" -v "$OUT:/out" -v ${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work:/rg:ro $NSMNT -w /work \
      thor:r38.3.arm64-sbsa-cu130-24.04-sglang_0.5.7 \
      bash -c "$NSYS profile $NFF -o /out/$TAG python3 /rg/nsys_sglang_guarded.py" >>"$LOG" 2>&1 ;;
  vllm)
    timeout 9000 docker run --rm --runtime=nvidia \
      -e BENCH_MODEL="$HFM" -e BENCH_PROMPT_LEN=128 -e BENCH_GEN_TOKENS=$G -e BENCH_NWARMUP=$NW \
      -e VLLM_ENFORCE_EAGER=1 -e BENCH_GPU_MEM=0.5 -e CPATH="$CP" \
      -v "$HF:/hf:ro" -v "$SCR:/work:ro" -v "$OUT:/out" $NSMNT -w /work \
      thor:r38.3.arm64-sbsa-cu130-24.04-vllm_0.12.0 \
      bash -c "$NSYS profile $NFC -o /out/$TAG python3 /work/scripts/nsys_vllm.py" >>"$LOG" 2>&1 ;;
  *) echo "unknown fw $FW"; exit 2 ;;
esac
RC=$?
echo "  docker exit=$RC rep=$([ -f $OUT/$TAG.nsys-rep ] && echo YES || echo NO) $(date)" | tee -a "$LOG"
if [ -f "$OUT/$TAG.nsys-rep" ]; then
  $NSYS export --type sqlite --force-overwrite=true -o "$OUT/$TAG.sqlite" "$OUT/$TAG.nsys-rep" >>"$LOG" 2>&1
  echo "  sqlite=$([ -f $OUT/$TAG.sqlite ] && echo YES || echo NO)" | tee -a "$LOG"
fi
tail -5 "$LOG"
