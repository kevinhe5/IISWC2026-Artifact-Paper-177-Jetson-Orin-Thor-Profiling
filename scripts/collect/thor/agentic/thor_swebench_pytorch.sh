#!/bin/bash
# Fig 12 (SWE-bench-live) on THOR — vLLM arm, adapted from orin/rerun_swebench_live_vllm.sh.
# Single container: vllm serve on :8000 + bench_swebench_live.py (real repos, real tools).
# Env: MODEL_REPO (default Llama-3.2-1B; use Qwen/Qwen3-4B for reasoning arm),
#      MODEL_TAG, NUM_TASKS (def 30), MAX_TURNS (def 30), CPUSET (optional ladder).
set -e
DATA=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data
BENCH_DIR="${DATA}/benchmarks/orin"                 # profiler code + tasks + repos
OUTD="${DATA}/benchmarks/thor/data/swebench"; mkdir -p "$OUTD"
TS=$(date +"%Y%m%d_%H%M%S")

CPUSET="${CPUSET:-}"
if [ -z "${CPUSET}" ]; then N_CORES=14; CPUSET_ARG=""
else N_CORES=$(echo "${CPUSET}" | awk -F- '{if(NF==1)print 1; else print $2-$1+1}'); CPUSET_ARG="--cpuset-cpus ${CPUSET}"; fi
MODEL_TAG="${MODEL_TAG:-1b}"
OUT_CSV="${OUTD}/pytorch_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}.csv"
LOG="${OUTD}/pytorch_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}.log"
TRACES="${OUTD}/pytorch_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}.traces.jsonl"
REPO_ROOT="${BENCH_DIR}/swebench_repos"
NUM_TASKS="${NUM_TASKS:-30}"; MAX_TURNS="${MAX_TURNS:-30}"; MAX_TOKENS="${MAX_TOKENS:-2048}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.5}"
TASKS_JSONL="${TASKS_JSONL:-${BENCH_DIR}/data/cache_workloads/swebench_live_30tasks.jsonl}"
CP="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include"
PT_IMG="thor:r38.3.arm64-sbsa-cu130-24.04-transformers"
HF_DIR="${DATA}/models/hf_full"
MODEL_REPO="${MODEL_REPO:-meta-llama/Llama-3.2-1B-Instruct}"
MODEL_DIR_NAME="models--${MODEL_REPO/\//--}"
SNAP=$(find "${HF_DIR}/${MODEL_DIR_NAME}/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
[ -z "${SNAP}" ] && { echo "ERROR: snapshot for ${MODEL_REPO} missing" >&2; exit 1; }
MODEL_CTR="/hf_models/${MODEL_DIR_NAME}/snapshots/$(basename "${SNAP}")"

[ -d "${REPO_ROOT}" ] || { echo "ERROR: ${REPO_ROOT} missing"; exit 1; }
[ -f "${TASKS_JSONL}" ] || { echo "ERROR: ${TASKS_JSONL} missing"; exit 1; }

# clock lock + assert; rg watchdog
sudo -n nvpmodel -m 0 >/dev/null 2>&1 || true; sudo -n jetson_clocks >/dev/null 2>&1 || true
G=$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq); E=$(cat /sys/class/devfreq/bwmgr/cur_freq)
[ "$G" = 1575000000 ] && [ "$E" = 4266000000 ] || { echo "!!! CLOCK NOT LOCKED — ABORT"; exit 1; }
( while sleep 5; do pkill -9 -x rg 2>/dev/null; done ) & WD=$!
trap 'kill $WD 2>/dev/null' EXIT
docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
sudo tee /proc/sys/vm/drop_caches <<<3 >/dev/null 2>&1; sleep 2

echo "[$(date)] THOR swebench live (vllm) model=${MODEL_REPO} tasks=${NUM_TASKS} turns=${MAX_TURNS} cores=${N_CORES}" | tee -a "${LOG}"

docker run --rm --runtime nvidia --network host ${CPUSET_ARG} \
    -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro \
    -e DEVICE_PROFILE=agx -e JETSON_PLATFORM=agx_thor_128gb -e HF_HOME=/hf_models \
    -v "${HF_DIR}:/hf_models" -v "${BENCH_DIR}:/benchmarks" -v "${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/scripts/agentic:/agentic" -v "${REPO_ROOT}:/repo" -v "${OUTD}:/out" \
    -e PYTHONPATH=/benchmarks/profiler_swebench_live:/benchmarks \
    -e AGENT_BASE_URL=http://127.0.0.1:8000/v1 -e AGENT_MODEL=agent \
    -e TASKS_JSONL=/benchmarks/data/cache_workloads/$(basename "${TASKS_JSONL}") -e REPO_ROOT=/repo \
    -e MAX_TURNS=${MAX_TURNS} -e MAX_TOKENS=${MAX_TOKENS} \
    -e DISABLE_STOP_TOKENS=${DISABLE_STOP_TOKENS:-0} -e PREFILL_THOUGHT=${PREFILL_THOUGHT:-0} \
    -e FRAMEWORK=pytorch -e STREAM_AGENT=1 -e QUANTIZATION=fp16 -e AG_THINK=${AG_THINK:-1} \
    -e CELL_LABEL=swebench_live_pytorch_thor \
    -e DUMP_TRACES=/out/$(basename "${TRACES}") \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CPATH="$CP" \
    "${PT_IMG}" \
    bash -c "
set -e
pip install -q --no-cache-dir pytest openai fastapi uvicorn 2>&1 | tail -1 || true
# NOTE: do NOT pip-install-editable the bug repos — a pydantic bug-variant shadows the
# venv's pydantic via its .pth and kills vllm serve at import. Tools (pytest/run_python)
# execute with cwd=instance dir, so the local package resolves without installation.
echo \"[container] transformers shim serving ${MODEL_CTR}\"
pip install -q fastapi uvicorn 2>&1 | tail -1; BENCH_MODEL=${MODEL_CTR} PORT=8000 BENCH_DTYPE=bf16 python3 /agentic/transformers_shim.py \
    > /out/pytorch_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}_server.log 2>&1 &
AGENT_PID=\$!
trap 'kill \$AGENT_PID 2>/dev/null || true' EXIT
for i in \$(seq 1 120); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo \"  shim up after \${i}*5s\"; break; }
    sleep 5
    [ \$i -eq 120 ] && { echo timeout; tail -30 /out/pytorch_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}_server.log; exit 1; }
done
python3 /benchmarks/profiler_swebench_live/bench_swebench_live.py /out/$(basename "${OUT_CSV}") ${NUM_TASKS}
" 2>&1 | tee -a "${LOG}"
echo "[$(date)] done; CSV rows: $(wc -l < "${OUT_CSV}" 2>/dev/null || echo 0), traces: $([ -f "${TRACES}" ] && wc -l < "${TRACES}" || echo 0)" | tee -a "${LOG}"
