#!/bin/bash
# Reviewer #1 (extension) — CV over the FULL quant/variant matrix, mirroring each master sweep
# script's exact bench invocation (paper-consistent). Shapes match the master:
#   llamacpp 6 quants  -> full 6x6 grid
#   vllm gguf 3 quants -> full 6x6 grid
#   llamacpp_fa/_fused -> f16 CROSS (11) + Q8_0/Q4_K_M/Q4_0 SINGLE(pp128_gen128)
#   sglang gguf 3      -> SINGLE ; pytorch 4bit/8bit -> SINGLE
# (trtedge quants handled by repeat_grid_trtedge_quant.sh — different image/runner.)
# Resume: cell with >=N rows SKIP'd. Order cheap->expensive (vllm gguf last, ~23h pole).
# SMOKE=1 -> N=1 and only pp128_gen128 per group, for a fast pipeline check.
set -u
N=${N:-3}
DATA=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data
HF=$DATA/models/hf_full; BENCH=$DATA/benchmarks/orin
CP=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include
SP=/opt/venv/lib/python3.12/site-packages
FUSED_WRAP=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/fused_compat/llama_cpp_fork
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

# ---- runners: each emits one JSON line, mirroring the matching master sweep script ----
llama_q(){    local q=$1 pp=$2 gen=$3 ctx=$(( $2+$3+128 ))
  docker run $DC -v "$DATA:/data" "$IMG_L" python3 /data/benchmarks/orin/profiler_llamacpp/bench_e2e.py \
    "/data/models/gguf/Llama-3.2-1B-Instruct-${q}.gguf" "$ctx" "$pp" "$gen" 99 "$pp" 1; }
llama_fa(){   local q=$1 pp=$2 gen=$3 ctx=$(( $2+$3+128 ))
  docker run $DC -e FLASH_ATTN=1 -v "$DATA:/data" "$IMG_L" python3 /data/benchmarks/orin/profiler_llamacpp/bench_e2e.py \
    "/data/models/gguf/Llama-3.2-1B-Instruct-${q}.gguf" "$ctx" "$pp" "$gen" 99 "$pp" 1; }
llama_fused(){ local q=$1 pp=$2 gen=$3 ctx=$(( $2+$3+128 ))
  docker run $DC -e FLASH_ATTN=1 -e LD_LIBRARY_PATH=${SP}/llama_cpp/lib -v "${FUSED_WRAP}:${SP}/llama_cpp:ro" -v "$DATA:/data" "$IMG_L" \
    python3 /data/benchmarks/orin/profiler_llamacpp/bench_e2e.py \
    "/data/models/gguf/Llama-3.2-1B-Instruct-${q}.gguf" "$ctx" "$pp" "$gen" 99 "$pp" 1; }
vllm_gguf(){  local q=$1 pp=$2 gen=$3
  docker run $DC -v "$DATA/models:/models" -v "$BENCH:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_vllm \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CPATH="$CP" "$IMG_V" \
    python3 /benchmarks/profiler_vllm/bench_e2e.py "/models/gguf/Llama-3.2-1B-Instruct-${q}.gguf" "$pp" "$gen" 1; }
sglang_gguf(){ local q=$1 pp=$2 gen=$3
  docker run $DC -v "$DATA/models:/models" -v "$BENCH:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_sglang \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CPATH=/usr/local/cuda/targets/sbsa-linux/include "$IMG_S" \
    python3 /benchmarks/profiler_sglang/bench_e2e.py "/models/gguf/Llama-3.2-1B-Instruct-${q}.gguf" "$pp" "$gen" 1; }
pytorch_q(){  local q=$1 pp=$2 gen=$3
  docker run $DC -v "$HF:/hf_models" -v "$BENCH:/benchmarks" -e PYTHONPATH=/benchmarks/profiler_pytorch \
    -e CPATH=/usr/local/cuda/targets/sbsa-linux/include "$IMG_P" \
    python3 /benchmarks/profiler_pytorch/bench_e2e.py "$HFM" "$pp" "$gen" "$q" 1; }

runN(){ local label=$1 fn=$2; shift 2; local f="$OUT/${label}.jsonl"
  local have=0; [ -f "$f" ] && have=$(grep -c . "$f" 2>/dev/null)
  if [ "$have" -ge "$N" ]; then echo "  [$label] SKIP ($have>=$N)"; return; fi
  for i in $(seq $((have+1)) $N); do clean; echo "  [$label] run $i/$N (have $have) $(date +%T)"
    local j; j=$("$fn" "$@" 2>/dev/null | grep '^{' | tail -1); [ -n "$j" ] && echo "$j">>"$f" || echo "    (empty $i)"
  done; echo "  -> $(wc -l <"$f") runs"; }

GRID="128 256 512 1024 2048 4096"
if [ "${SMOKE:-0}" = 1 ]; then N=1; GRID="128"; fi
cross_cells(){ for pp in $GRID; do echo "$pp 128"; done; for gen in 256 512 1024 2048 4096; do echo "128 $gen"; done; }
[ "${SMOKE:-0}" = 1 ] && cross_cells(){ echo "128 128"; }

echo "######## REPEAT GRID QUANT N=$N $(date) ########"
# 1) pytorch quant singles (fast)
for q in 8bit 4bit; do runN "pytorch_eager_${q}_pp128_gen128" pytorch_q "$q" 128 128; done
# 2) sglang gguf singles
for q in gguf_Q8_0 gguf_Q4_K_M gguf_Q4_0; do runN "sglang_${q}_pp128_gen128" sglang_gguf "${q#gguf_}" 128 128; done
# 3) llamacpp_fa : f16 cross + quant singles
echo "==== llamacpp_fa ===="
while read pp gen; do runN "llamacpp_fa_f16_pp${pp}_gen${gen}" llama_fa f16 "$pp" "$gen"; done < <(cross_cells)
for q in Q8_0 Q4_K_M Q4_0; do runN "llamacpp_fa_${q}_pp128_gen128" llama_fa "$q" 128 128; done
# 4) llamacpp_fused : f16 cross + quant singles
echo "==== llamacpp_fused ===="
while read pp gen; do runN "llamacpp_fused_f16_pp${pp}_gen${gen}" llama_fused f16 "$pp" "$gen"; done < <(cross_cells)
for q in Q8_0 Q4_K_M Q4_0; do runN "llamacpp_fused_${q}_pp128_gen128" llama_fused "$q" 128 128; done
# 5) llamacpp quant ladder : full 6x6 (fast per run)
for q in Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_L; do echo "==== llamacpp $q ===="
  for pp in $GRID; do for gen in $GRID; do runN "llama_${q}_pp${pp}_gen${gen}" llama_q "$q" "$pp" "$gen"; done; done; done
# 6) vllm gguf : full 6x6 (SLOW ~250s/run — the pole, runs last)
for q in Q8_0 Q4_K_M Q4_0; do echo "==== vllm gguf_$q ===="
  for pp in $GRID; do for gen in $GRID; do runN "vllm_gguf_${q}_pp${pp}_gen${gen}" vllm_gguf "$q" "$pp" "$gen"; done; done; done
echo "######## DONE $(date) ########"
