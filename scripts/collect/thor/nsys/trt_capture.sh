#!/bin/bash
# TRT-Edge-LLM decode nsys capture (Fig7 fp16 column). llm_bench decode, osl=128 =>
# 128 real decode tokens at pastKVLen=128. Engine chosen by ENG arg (eng_host = fp16).
# Usage: trt_capture.sh <engine_subdir> <tag>
set -u
ENG="$1"; TAG="$2"
REPO="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/TensorRT-Edge-LLM"
HLIB="/usr/lib/aarch64-linux-gnu"; CULIB="/usr/local/cuda/targets/sbsa-linux/lib"
RUNIMG="thor:r38.3.arm64-sbsa-cu130-24.04-transformers"
NSROOT="/opt/nvidia/nsight-systems/2025.6.3"; NSYS="$NSROOT/target-linux-sbsa-armv8/nsys"; NSMNT="-v $NSROOT:$NSROOT:ro"
NFT="--trace=cuda --cuda-graph-trace=node --sample=none --force-overwrite=true"
OUT="${NSYS_TRACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/data/nsys/traces}"
LOG="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work/logs/cap_${TAG}.log"
LD="LD_LIBRARY_PATH=/hl:/hcuda:/repo/build:\$LD_LIBRARY_PATH EDGELLM_PLUGIN_PATH=/repo/build/libNvInfer_edgellm_plugin.so"

pkill -9 -x rg 2>/dev/null; docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
sudo jetson_clocks >/dev/null 2>&1 || true; sudo tee /proc/sys/vm/drop_caches <<<3 >/dev/null 2>&1 || true
GC=$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq); [ "$GC" = 1575000000 ] || { echo "CLOCK UNLOCKED $GC ABORT"; exit 1; }; sleep 2

echo "######## TRT capture ENG=$ENG tag=$TAG $(date) ########" > "$LOG"
timeout 900 docker run --rm --runtime=nvidia -v "$REPO:/repo" -v "$HLIB:/hl:ro" -v "$CULIB:/hcuda:ro" \
  -v "$OUT:/out" $NSMNT -w /repo "$RUNIMG" \
  bash -c "$LD $NSYS profile $NFT -o /out/$TAG \
    /repo/build/examples/llm/llm_bench --engineDir /repo/workspace/$ENG --mode decode \
    --inputLen 1 --pastKVLen 128 --osl 128 --useCudaGraph --noProfile --iterations 1 --warmup 3" >>"$LOG" 2>&1
RC=$?
echo "  docker exit=$RC rep=$([ -f $OUT/$TAG.nsys-rep ] && echo YES || echo NO)" | tee -a "$LOG"
if [ -f "$OUT/$TAG.nsys-rep" ]; then
  $NSYS export --type sqlite --force-overwrite=true -o "$OUT/$TAG.sqlite" "$OUT/$TAG.nsys-rep" >>"$LOG" 2>&1
  echo "  sqlite=$([ -f $OUT/$TAG.sqlite ] && echo YES || echo NO)" | tee -a "$LOG"
fi
grep -aiE "exit=|sqlite=|Per-step|Tokens/sec|E2E|error" "$LOG" | tail -6
