#!/bin/bash
# Reviewer #1 (extension) — TRT-Edge-LLM repeatability, paper-consistent runner.
# TRT-Edge measures separably (this is how the paper's trtedge_llm grid was built, trt_ttft_tpot.sh):
#   prefill --inputLen I   -> "E2E Time"  = TTFT(pp axis)
#   decode  --pastKVLen P  -> "Per-step"  = TPOT(context axis)
# The 6x6 grid is the outer product of these axes, so full pp/gen coverage = the 6 prefill + 6 decode
# axis points. Here each axis point is run N independent times (fresh docker + drop_caches + relock)
# so mean/std/CV can be computed with the SAME llm_bench flags as the master sweep.
set -u
N=${N:-3}
REPO=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/TensorRT-Edge-LLM; WORK=$REPO/workspace
HLIB=/usr/lib/aarch64-linux-gnu; CULIB=/usr/local/cuda/targets/sbsa-linux/lib
IMG=thor:r38.3.arm64-sbsa-cu130-24.04-transformers
LD='LD_LIBRARY_PATH=/hl:/hcuda:/repo/build:$LD_LIBRARY_PATH EDGELLM_PLUGIN_PATH=/repo/build/libNvInfer_edgellm_plugin.so'
ENG=/repo/workspace/eng_trt_sweep          # maxInputLen=4096, KV=9216 (covers inputLen<=4096, pastKVLen<=4096)
OUT=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/data/repeat_stats_trtedge; mkdir -p "$OUT"
LOG=${LOG_DIR:-/nvme/iiswc}/trtedge_repeat_$(date +%Y%m%d_%H%M%S).log; ln -sf "$LOG" ${LOG_DIR:-/nvme/iiswc}/trtedge_repeat_latest.log

( while :; do for p in $(pgrep -x rg); do kill -STOP "$p" 2>/dev/null; done; sleep 0.3; done ) & WD=$!
trap "kill $WD 2>/dev/null" EXIT
clean(){ docker ps -aq|xargs -r docker rm -f >/dev/null 2>&1; sudo -n jetson_clocks >/dev/null 2>&1||true
  G=$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq); [ "$G" = 1575000000 ] || echo "  WARN gpu clock=$G"
  sudo -n tee /proc/sys/vm/drop_caches <<<3 >/dev/null 2>&1||true; sleep 1; }
dock(){ timeout 600 docker run --rm --runtime=nvidia -v "$REPO:/repo" -v "$HLIB:/hl:ro" -v "$CULIB:/hcuda:ro" -w /repo "$IMG" bash -c "$LD $*"; }

echo "######## TRTEDGE REPEAT N=$N $(date) ########" | tee "$LOG"
# Build the sweep engine once if absent (same params as trt_ttft_tpot.sh).
if [ ! -f "$WORK/eng_trt_sweep/llm.engine" ]; then
  clean; echo "#### build eng_trt_sweep (maxInputLen=4096 KV=9216) ####" | tee -a "$LOG"
  dock "/repo/build/examples/llm/llm_build --onnxDir /repo/workspace/Llama-3.2-1B/onnx --engineDir /repo/workspace/eng_trt_sweep --maxInputLen 4096 --maxKVCacheCapacity 9216" >>"$LOG" 2>&1
  for f in config.json tokenizer.json tokenizer_config.json embedding.safetensors processed_chat_template.json; do cp -n "$WORK/eng_host/$f" "$WORK/eng_trt_sweep/" 2>/dev/null; done
fi
[ -f "$WORK/eng_trt_sweep/llm.engine" ] || { echo "BUILD FAILED — no engine"; exit 1; }

repeatN(){  # label  metric_key  bench_args...   ; parses E2E (prefill) or Per-step (decode)
  local label=$1 mk=$2; shift 2; local f="$OUT/${label}.jsonl"
  local have=0; [ -f "$f" ] && have=$(grep -c . "$f" 2>/dev/null)
  if [ "$have" -ge "$N" ]; then echo "  [$label] SKIP ($have>=$N)"; return; fi
  for i in $(seq $((have+1)) $N); do clean; echo "  [$label] run $i/$N (have $have) $(date +%T)"
    local out val; out=$(dock "$@" 2>&1)
    # This llm_bench build reports both prefill TTFT and decode TPOT on the same line
    # ("E2E Time (actual performance): X ms"); decode uses OSL=1 so E2E == per-step TPOT.
    val=$(echo "$out" | grep -aoE 'E2E Time \(actual performance\): [0-9.]+' | grep -oE '[0-9.]+' | tail -1)
    [ -n "$val" ] && echo "{\"$mk\": $val}" >>"$f" || echo "    (empty $i)"
  done; echo "  -> $(wc -l <"$f") runs"; }

echo "==== prefill / TTFT vs inputLen ===="
for I in 128 256 512 1024 2048 4096; do
  repeatN "trtedge_prefill_isl${I}" ttft_ms \
    "/repo/build/examples/llm/llm_bench --engineDir $ENG --mode prefill --inputLen $I --useCudaGraph --noProfile --iterations 10 --warmup 3"
done
echo "==== decode / TPOT vs pastKVLen ===="
for P in 128 256 512 1024 2048 4096; do
  repeatN "trtedge_decode_ctx${P}" tpot_ms \
    "/repo/build/examples/llm/llm_bench --engineDir $ENG --mode decode --inputLen 1 --pastKVLen $P --osl 1 --useCudaGraph --noProfile --iterations 20 --warmup 5"
done
echo "######## DONE $(date) ########" | tee -a "$LOG"
# CV summary
python3 - "$OUT" <<'PY'
import glob,json,os,sys,statistics as st
D=sys.argv[1]
for f in sorted(glob.glob(f"{D}/*.jsonl")):
    v=[list(json.loads(l).values())[0] for l in open(f) if l.strip()]
    if len(v)<2: print(f"  {os.path.basename(f):30} n={len(v)} {v}"); continue
    m=st.mean(v); cv=st.pstdev(v)/m*100
    print(f"  {os.path.basename(f):30} mean {m:8.3f}  cv {cv:4.2f}%  n={len(v)}")
PY
