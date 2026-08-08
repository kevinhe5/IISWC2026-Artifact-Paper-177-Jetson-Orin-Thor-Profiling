#!/usr/bin/env bash
# TensorRT Edge-LLM 0.7.0 sweep for Jetson AGX Thor (framework="trtedge_llm").
# Built from source for sm_110a (NVIDIA/TensorRT-Edge-LLM release/0.7.0 = paper's TRT-Edge-LLM 0.7).
# Uses the native llm_bench tool (isolated prefill/decode microbench = paper Fig5/Fig6 methodology):
#   prefill mode --inputLen N      -> TTFT vs ISL
#   decode  mode --pastKVLen N     -> TPOT vs effective context
# Parses "E2E Time (actual performance): X ms". Clocks locked @ MAXN.
set -uo pipefail
EDGE=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/TensorRT-Edge-LLM
IMG=trtedge_img2:latest
ENG=/src/workspace/Llama-3.2-1B/engines
OUTDIR=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/data/sweep_results_agx_thor_128gb
mkdir -p "$OUTDIR"
CSV="${OUTPUT_CSV:-$OUTDIR/sweep_thor_trtedge_$(date +%Y%m%d_%H%M%S).csv}"
BENCH=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/orin

source "$BENCH/_lock_clocks.sh" && lock_clocks "$OUTDIR/clocks.log" || sudo -n /usr/bin/jetson_clocks

[ -s "$CSV" ] || echo "timestamp,framework,model,phase,length,e2e_ms,tps,mha_ms,gemm_ms,other_ms" > "$CSV"

run() {  # phase lenflag len
  local phase="$1" flag="$2" len="$3"
  echo "  [run] trtedge $phase $len"
  local out; out=$(docker run --rm --runtime nvidia -v "$EDGE:/src" -w /src "$IMG" \
     ./build/examples/llm/llm_bench --engineDir "$ENG" --mode "$phase" "$flag" "$len" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  local e2e tps mha gemm other
  e2e=$(echo "$out" | grep -oE "E2E Time \(actual performance\): [0-9.]+ ms" | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
  tps=$(echo "$out" | grep -oE "Tokens/sec \(E2E\): [0-9.]+" | sed -E 's/.*: //' | head -1)
  mha=$(echo "$out" | grep -oE "MHA: +[0-9.]+" | grep -oE "[0-9.]+" | head -1)
  gemm=$(echo "$out" | grep -oE "GEMM: +[0-9.]+" | grep -oE "[0-9.]+" | head -1)
  other=$(echo "$out" | grep -oE "Kgen\+Other: +[0-9.]+" | grep -oE "[0-9.]+" | head -1)
  if [ -n "$e2e" ]; then
    echo "$(date '+%Y%m%d %H:%M:%S'),trtedge_llm,Llama-3.2-1B,$phase,$len,$e2e,${tps:-0},${mha:-0},${gemm:-0},${other:-0}" >> "$CSV"
    echo "    -> ${e2e}ms ${tps:-} tps"
  else
    echo "    SKIP: no E2E result"; echo "$out" | tail -3
  fi
}

echo "===== TRT-Edge-LLM Thor sweep -> $CSV ====="
echo "--- prefill (TTFT vs ISL) ---"
for n in 128 256 512 1024 2048 4096; do run prefill --inputLen "$n"; done
echo "--- decode (TPOT vs context) ---"
for n in 128 256 512 1024 2048 4096; do run decode --pastKVLen "$n"; done
echo "===== DONE: $(($(wc -l < "$CSV")-1)) rows in $CSV ====="
