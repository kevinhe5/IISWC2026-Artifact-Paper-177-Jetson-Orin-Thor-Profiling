#!/bin/bash
set -u
HF="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/models/hf_full"; WORK="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/orin/nsys_profiles"
HFM="/hf/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
CP="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include"; IMG_P="thor:r38.3.arm64-sbsa-cu130-24.04-transformers"
PTXFIX='cp -f /usr/local/cuda/bin/ptxas /opt/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas-blackwell 2>/dev/null; cp -f /usr/local/cuda/bin/ptxas /opt/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas 2>/dev/null'
LOG=${LOG_DIR:-/nvme/iiswc}/mbu_pytorch_thor.log
clean(){ pkill -9 -x rg 2>/dev/null; docker ps -aq|xargs -r docker rm -f >/dev/null 2>&1; sudo jetson_clocks >/dev/null 2>&1; sudo tee /proc/sys/vm/drop_caches <<<3 >/dev/null 2>&1
  GC=$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq); [ "$GC" = 1575000000 ]||{ echo CLOCK_ABORT;exit 1;}; sleep 1; }
tpot(){ grep -aoE 'median tpot=[0-9.]+' "$1"|tail -1|grep -oE '[0-9.]+'; }
echo "## MBU pytorch Thor pp512/gen256 $(date)">"$LOG"
clean; echo "#### eager (pure) ####"|tee -a "$LOG"; T=/tmp/mbu_eager.log
timeout 1200 docker run --rm --runtime=nvidia -e BENCH_MODEL="$HFM" -e PYTORCH_ATTN_IMPL=eager -e BENCH_DTYPE=bf16 -e BENCH_PROMPT_LEN=512 -e BENCH_GEN_TOKENS=256 -e BENCH_WARMUP=3 -e BENCH_REPEATS=5 -e CPATH="$CP" -v "$HF:/hf:ro" -v "$WORK:/work" -w /work "$IMG_P" python3 /work/scripts/bench_pytorch_thor.py >"$T" 2>&1
echo "pytorch_eager pp512 gen256 TPOT=$(tpot $T)"|tee -a "$LOG"
clean; echo "#### compile (sdpa) ####"|tee -a "$LOG"; T=/tmp/mbu_compile.log
timeout 1800 docker run --rm --runtime=nvidia -e BENCH_MODEL="$HFM" -e PYTORCH_ATTN_IMPL=sdpa -e BENCH_COMPILE=1 -e BENCH_DTYPE=bf16 -e BENCH_PROMPT_LEN=512 -e BENCH_GEN_TOKENS=256 -e BENCH_WARMUP=3 -e BENCH_REPEATS=5 -e CPATH="$CP" -v "$HF:/hf:ro" -v "$WORK:/work" -w /work "$IMG_P" bash -c "$PTXFIX; python3 /work/scripts/bench_pytorch_thor.py" >"$T" 2>&1
echo "pytorch_compile pp512 gen256 TPOT=$(tpot $T)"|tee -a "$LOG"
echo "## DONE $(date)"|tee -a "$LOG"
