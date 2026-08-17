#!/bin/bash
# Live SWE-style code-agent — SGLang single-engine on :8000.
set -e
DATA="${DATA_ROOT:-${PROFILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/profile}}"
BENCH_DIR="${DATA}/benchmarks"
TS=$(date +"%Y%m%d_%H%M%S")

# CPU ladder support (Plan A §III.C extension):
#   CPUSET=""        → unpinned (all 12 cores; default if unset)
#   CPUSET="0"       → 1 core
#   CPUSET="0-3"     → 4 cores
#   CPUSET="0-7"     → 8 cores
CPUSET="${CPUSET:-}"
if [ -z "${CPUSET}" ]; then
    N_CORES=12
    CPUSET_ARG=""
else
    N_CORES=$(echo "${CPUSET}" | awk -F- '{if(NF==1)print 1; else print $2-$1+1}')
    CPUSET_ARG="--cpuset-cpus ${CPUSET}"
fi
MODEL_TAG="${MODEL_TAG:-1b}"
OUT_CSV="${BENCH_DIR}/sweep_results/sglang_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}.csv"
LOG="${BENCH_DIR}/sweep_results/sglang_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}.log"
TRACES="${BENCH_DIR}/sweep_results/sglang_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}.traces.jsonl"
AGENT_LOG="${BENCH_DIR}/sweep_results/sglang_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}_server_agent.log"
REPO_ROOT="${BENCH_DIR}/swebench_repos"
NUM_TASKS="${NUM_TASKS:-5}"
MAX_TURNS="${MAX_TURNS:-15}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

[ ! -d "${REPO_ROOT}" ] && { echo "ERROR: REPO_ROOT ${REPO_ROOT} missing — run prepare_swebench_repos.sh first" >&2; exit 1; }
TASKS_JSONL="${TASKS_JSONL:-${BENCH_DIR}/data/cache_workloads/swebench_live_5tasks.jsonl}"
[ ! -f "${TASKS_JSONL}" ] && { echo "ERROR: ${TASKS_JSONL} missing" >&2; exit 1; }
python3 - <<PYEOF || exit 1
import json, os, sys
missing = []
for line in open("${TASKS_JSONL}"):
    iid = (json.loads(line).get("instance_id") or "").strip()
    if iid and not os.path.isdir(os.path.join("${REPO_ROOT}", iid)):
        missing.append(iid)
if missing:
    sys.stderr.write(f"ERROR: {len(missing)} repo dir(s) missing under ${REPO_ROOT}: {missing[:3]}\\n")
    sys.stderr.write("Run: TASKS_JSONL=${TASKS_JSONL} bash prepare_swebench_repos.sh\\n")
    sys.exit(1)
PYEOF
running=$(docker ps -q | wc -l)
[ "$running" -gt 0 ] && { echo "ERROR: container running" | tee -a "${LOG}"; exit 1; }

HF_DIR="${DATA}/models/hf_full"
MODEL_REPO="${MODEL_REPO:-meta-llama/Llama-3.2-1B-Instruct}"
MODEL_DIR_NAME="models--${MODEL_REPO/\//--}"
LLAMA_SNAP=$(find "${HF_DIR}/${MODEL_DIR_NAME}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
[ -z "${LLAMA_SNAP}" ] && { echo "ERROR: HF snapshot for ${MODEL_REPO} missing" >&2; exit 1; }
LLAMA_CTR="/hf_models/${MODEL_DIR_NAME}/snapshots/$(basename ${LLAMA_SNAP})"
echo "[$(date)] sglang model = ${MODEL_REPO}" | tee -a "${LOG}"
SGLANG_IMG="sglang-orin:0.4.6-sm87"

echo "[$(date)] swebench live (sglang) launching" | tee -a "${LOG}"
docker run --rm --runtime nvidia --network host ${CPUSET_ARG} \
    -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /sys:/sys:ro \
    -v /proc/device-tree:/proc/device-tree:ro -e DEVICE_PROFILE=agx \
    -v "${HF_DIR}:/hf_models" -v "${BENCH_DIR}:/benchmarks" -v "${REPO_ROOT}:/repo" \
    -e PYTHONPATH=/benchmarks/profiler_swebench_live:/benchmarks \
    -e AGENT_BASE_URL=http://127.0.0.1:8000/v1 -e AGENT_MODEL=agent \
    -e TASKS_JSONL=/benchmarks/data/cache_workloads/$(basename ${TASKS_JSONL}) -e REPO_ROOT=/repo -e MAX_TURNS=${MAX_TURNS} -e MAX_TOKENS=${MAX_TOKENS} -e ONLY_TASKS=${ONLY_TASKS:-} -e PREFILL_THOUGHT=${PREFILL_THOUGHT:-0} -e THINK_OFF=${THINK_OFF:-0} -e FRAMEWORK=sglang \
    -e STREAM_AGENT=${STREAM_AGENT:-1} -e QUANTIZATION=fp16 -e CELL_LABEL=swebench_live_sglang \
    -e DUMP_TRACES=/benchmarks/sweep_results/$(basename ${TRACES}) \
    "${SGLANG_IMG}" \
    bash -c "
set -e
pip install -q --no-cache-dir --index-url https://pypi.org/simple/ openai 2>&1 | tail -3 || pip install -q --no-cache-dir openai 2>&1 | tail -3
echo '[container] starting sglang on :8000…'
pip install -q --no-cache-dir --index-url https://pypi.org/simple/ pytest 2>&1 | tail -2 || pip install -q --no-cache-dir pytest 2>&1 | tail -2
# Each bug repo is pure-Python; install editably so pytest can import them.
for IID_DIR in /repo/*/; do
    [ -f "\$IID_DIR/setup.py" ] || [ -f "\$IID_DIR/pyproject.toml" ] || continue
    pip install -q --no-deps -e "\$IID_DIR" 2>&1 | tail -1 || true
done
export PYTHONPATH=/repo:${PYTHONPATH}
# Pick tool-call-parser based on model family.
TC_PARSER=\"llama3\"
REASONING_ARG=\"\"
case \"${MODEL_REPO}\" in
    *Qwen3*|*qwen3*) TC_PARSER=\"qwen25\"; REASONING_ARG=\"--reasoning-parser qwen3\" ;;
esac
echo \"[container] sglang tool-call-parser=\${TC_PARSER} reasoning=\${REASONING_ARG:-off}\"
python3 -m sglang.launch_server \
    --model-path ${LLAMA_CTR} \
    --served-model-name agent \
    --host 127.0.0.1 --port 8000 \
    --dtype float16 \
    --mem-fraction-static ${SGLANG_MEM_FRAC:-0.55} \
    --context-length ${SGLANG_CONTEXT:-32768} \
    --chunked-prefill-size ${SGLANG_CHUNK:-8192} \
    --tool-call-parser \${TC_PARSER} \
    \${REASONING_ARG} \
    --grammar-backend outlines \
    --attention-backend triton \
    --disable-cuda-graph \
    > /benchmarks/sweep_results/$(basename ${AGENT_LOG}) 2>&1 &
AGENT_PID=\$!
trap 'kill \$AGENT_PID 2>/dev/null || true' EXIT
for i in \$(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then echo \"  port 8000 up after \${i}*5s\"; break; fi
    sleep 5
    [ \$i -eq 60 ] && { echo '  timeout'; tail -60 /benchmarks/sweep_results/$(basename ${AGENT_LOG}); exit 1; }
done
python3 /benchmarks/profiler_swebench_live/bench_swebench_live.py \
    /benchmarks/sweep_results/$(basename ${OUT_CSV}) ${NUM_TASKS}
" 2>&1 | tee -a "${LOG}"

echo "[$(date)] done; CSV: ${OUT_CSV} rows=$(wc -l < ${OUT_CSV} 2>/dev/null || echo 0)" | tee -a "${LOG}"
