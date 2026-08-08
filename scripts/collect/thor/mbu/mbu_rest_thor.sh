#!/bin/bash
# Measure remaining Thor MBU bars at pp512/gen256 (T=640): vLLM, SGLang, llama.cpp(FA off), TRT-Edge.
set -u
HF="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/models/hf_full"; GGUF="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/models/gguf"
WORK="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/orin/nsys_profiles"; REPO="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/TensorRT-Edge-LLM"
HFM="/hf/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
CP="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include"; HLIB="/usr/lib/aarch64-linux-gnu"; CULIB="/usr/local/cuda/targets/sbsa-linux/lib"
IMG_V="thor:r38.3.arm64-sbsa-cu130-24.04-vllm_0.12.0"; IMG_S="thor:r38.3.arm64-sbsa-cu130-24.04-sglang_0.5.7"
IMG_L="thor:r38.3.arm64-sbsa-cu130-24.04-llama_cpp_b5255"; IMG_T="thor:r38.3.arm64-sbsa-cu130-24.04-transformers"
LD="LD_LIBRARY_PATH=/hl:/hcuda:/repo/build:\$LD_LIBRARY_PATH EDGELLM_PLUGIN_PATH=/repo/build/libNvInfer_edgellm_plugin.so"
PATCH='sed -i "/^            kv_data_type,\$/a\\            None,  # o_data_type" /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flashinfer.py'
LOG=${LOG_DIR:-/nvme/iiswc}/mbu_rest_thor.log
clean(){ pkill -9 -x rg 2>/dev/null; docker ps -aq|xargs -r docker rm -f >/dev/null 2>&1; sudo jetson_clocks >/dev/null 2>&1; sudo tee /proc/sys/vm/drop_caches <<<3 >/dev/null 2>&1
  GC=$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq); [ "$GC" = 1575000000 ]||{ echo CLOCK_ABORT;exit 1;}; sleep 1; }
tpot(){ grep -aoE 'median tpot=[0-9.]+' "$1"|tail -1|grep -oE '[0-9.]+'; }
echo "## MBU REST Thor pp512/gen256 $(date)">"$LOG"
clean; echo "#### vLLM ####"|tee -a "$LOG"; T=/tmp/mr_vllm.log
timeout 1200 docker run --rm --runtime=nvidia -e BENCH_MODEL="$HFM" -e BENCH_QUANT=none -e BENCH_PROMPT_LEN=512 -e BENCH_GEN_TOKENS=256 -e BENCH_WARMUP=3 -e BENCH_REPEATS=5 -e VLLM_ENFORCE_EAGER=1 -e VLLM_ASYNC_SCHEDULING=0 -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_ATTENTION_BACKEND=FLASHINFER -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_DISABLE_PREFIX_CACHE=1 -e CPATH="$CP" -v "$HF:/hf:ro" -v "$WORK:/work" -w /work "$IMG_V" bash -c "$PATCH; python3 /work/scripts/bench_vllm_thor.py" >"$T" 2>&1
echo "vllm pp512 gen256 TPOT=$(tpot $T)"|tee -a "$LOG"
clean; echo "#### SGLang ####"|tee -a "$LOG"; T=/tmp/mr_sglang.log
timeout 1200 docker run --rm --runtime=nvidia -e BENCH_MODEL="$HFM" -e BENCH_PROMPT_LEN=512 -e BENCH_GEN_TOKENS=256 -e BENCH_WARMUP=3 -e BENCH_REPEATS=5 -e BENCH_ATTN_BACKEND=flashinfer -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CPATH="$CP" -v "$HF:/hf:ro" -v "$WORK:/work" -w /work "$IMG_S" python3 /work/scripts/bench_sglang_thor.py >"$T" 2>&1
echo "sglang pp512 gen256 TPOT=$(tpot $T)"|tee -a "$LOG"
clean; echo "#### llama.cpp FA off ####"|tee -a "$LOG"; T=/tmp/mr_llama.log
timeout 1200 docker run --rm --runtime=nvidia -e BENCH_MODEL="/gguf/Llama-3.2-1B-Instruct-f16.gguf" -e BENCH_FLASH=0 -e BENCH_PROMPT_LEN=512 -e BENCH_GEN_TOKENS=256 -e BENCH_WARMUP=3 -e BENCH_REPEATS=5 -v "$GGUF:/gguf:ro" -v "$WORK:/work" -w /work "$IMG_L" python3 /work/scripts/bench_llamacpp_thor.py >"$T" 2>&1
echo "llama pp512 gen256 TPOT=$(tpot $T)"|tee -a "$LOG"
clean; echo "#### TRT-Edge decode pastKVLen=640 ####"|tee -a "$LOG"; T=/tmp/mr_trt.log
docker run --rm --runtime=nvidia -v "$REPO:/repo" -v "$HLIB:/hl:ro" -v "$CULIB:/hcuda:ro" -w /repo "$IMG_T" bash -c "$LD /repo/build/examples/llm/llm_bench --engineDir /repo/workspace/eng_trt_sweep --mode decode --inputLen 1 --pastKVLen 640 --osl 1 --useCudaGraph --noProfile --iterations 20 --warmup 5" >"$T" 2>&1
PS=$(grep -aioE 'Per-step[^0-9]*[0-9.]+' "$T"|grep -oE '[0-9.]+'|tail -1); DE=$(grep -aioE 'Decode E2E Time:[^0-9]*[0-9.]+' "$T"|grep -oE '[0-9.]+'|tail -1)
echo "trt pastKVLen=640 Per-step=$PS Decode-E2E=$DE"|tee -a "$LOG"
echo "## MBU REST DONE $(date)"|tee -a "$LOG"
