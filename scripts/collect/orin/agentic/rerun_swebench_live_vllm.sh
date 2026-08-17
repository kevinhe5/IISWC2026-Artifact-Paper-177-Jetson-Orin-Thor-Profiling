#!/bin/bash
# Live SWE-style code-agent across vLLM. SINGLE engine on :8000 (no
# user-sim — the bug report is the static prompt). Real Python repo
# mounted to /repo, agent uses real tools (view/list/grep/edit/run_python).
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
# Derive a short model tag for filenames: "1b" / "3b" / "8b" (default "1b").
MODEL_TAG="${MODEL_TAG:-1b}"
OUT_CSV="${BENCH_DIR}/sweep_results/vllm_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}.csv"
LOG="${BENCH_DIR}/sweep_results/vllm_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}.log"
TRACES="${BENCH_DIR}/sweep_results/vllm_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}.traces.jsonl"

# Test repo: tau-bench source (already on disk, real Python project ~50 files)
REPO_ROOT="${BENCH_DIR}/swebench_repos"

NUM_TASKS="${NUM_TASKS:-5}"
MAX_TURNS="${MAX_TURNS:-15}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
# vLLM GPU-memory utilization. 0.55 (~16 GB on 32 GB SoC) is fine for 1B/3B fp16;
# 8B fp16 weights alone are ~16 GB so we need ~0.85.
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.55}"

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
[ "$running" -gt 0 ] && { echo "ERROR: container(s) running" | tee -a "${LOG}"; exit 1; }

echo "[$(date)] swebench live (vllm) launching" | tee -a "${LOG}"
sudo -n /usr/bin/jetson_clocks --show 2>&1 | grep -E "GPU|EMC|Power" | head -3 | tee -a "${LOG}"

VLLM_IMG="dustynv/vllm:0.8.6-r36.4-cu128-24.04"
HF_DIR="${DATA}/models/hf_full"
# MODEL_REPO env: pick "Llama-3.2-1B-Instruct" (default) / "Llama-3.2-3B-Instruct" /
# "Llama-3.1-8B-Instruct" — must exist under ${HF_DIR}/models--meta-llama--<name>/snapshots/.
MODEL_REPO="${MODEL_REPO:-meta-llama/Llama-3.2-1B-Instruct}"
MODEL_DIR_NAME="models--${MODEL_REPO/\//--}"
LLAMA_SNAP=$(find "${HF_DIR}/${MODEL_DIR_NAME}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
[ -z "${LLAMA_SNAP}" ] && { echo "ERROR: HF snapshot for ${MODEL_REPO} missing at ${HF_DIR}/${MODEL_DIR_NAME}/snapshots" >&2; exit 1; }
LLAMA_CTR="/hf_models/${MODEL_DIR_NAME}/snapshots/$(basename ${LLAMA_SNAP})"
echo "[$(date)] vllm model = ${MODEL_REPO} (snapshot $(basename ${LLAMA_SNAP}))" | tee -a "${LOG}"

# Single dual-purpose container: starts vLLM serve, waits for it, runs the bench.
docker run --rm --runtime nvidia --network host ${CPUSET_ARG} \
    -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro \
    -v /sys:/sys:ro \
    -v /proc/device-tree:/proc/device-tree:ro \
    -e DEVICE_PROFILE=agx \
    -e HF_HOME=/hf_models \
    -v "${HF_DIR}:/hf_models" \
    -v "${BENCH_DIR}:/benchmarks" \
    -v "${REPO_ROOT}:/repo" \
    -e PYTHONPATH=/benchmarks/profiler_swebench_live:/benchmarks \
    -e AGENT_BASE_URL=http://127.0.0.1:8000/v1 \
    -e AGENT_MODEL=agent \
    -e TASKS_JSONL=/benchmarks/data/cache_workloads/$(basename ${TASKS_JSONL}) -e REPO_ROOT=/repo \
    -e MAX_TURNS=${MAX_TURNS} \
    -e MAX_TOKENS=${MAX_TOKENS} \
    -e ONLY_TASKS=${ONLY_TASKS:-} \
    -e DISABLE_STOP_TOKENS=${DISABLE_STOP_TOKENS:-0} \
    -e PREFILL_THOUGHT=${PREFILL_THOUGHT:-0} \
    -e THINK_OFF=${THINK_OFF:-0} \
    -e FRAMEWORK=vllm \
    -e STREAM_AGENT=1 \
    -e QUANTIZATION=fp16 \
    -e CELL_LABEL=swebench_live_vllm \
    -e DUMP_TRACES=/benchmarks/sweep_results/$(basename ${TRACES}) \
    "${VLLM_IMG}" \
    bash -c "
set -e
echo '[container] installing pytest…'
pip install -q --no-cache-dir --index-url https://pypi.org/simple/ pytest 2>&1 | tail -2 || pip install -q --no-cache-dir pytest 2>&1 | tail -2
# Each bug repo is pure-Python; install editably so pytest can import them.
for IID_DIR in /repo/*/; do
    [ -f "\$IID_DIR/setup.py" ] || [ -f "\$IID_DIR/pyproject.toml" ] || continue
    pip install -q --no-deps -e "\$IID_DIR" 2>&1 | tail -1 || true
done
# Make /repo importable so 'import tau_bench' works inside pytest subprocess
export PYTHONPATH=/repo:\${PYTHONPATH}
echo '[container] starting vLLM agent server on :8000…'
# Qwen3 emits <think>...</think> reasoning + <tool_call>...</tool_call> in
# Hermes format. Llama-3 uses <|python_tag|>{json} + llama3_json parser.
# Pick parsers based on MODEL_REPO substring. The HOST already substituted
# MODEL_REPO so the case below sees the literal string at container time.
TC_PARSER=\"llama3_json\"
REASONING_ARGS=\"\"
case \"${MODEL_REPO}\" in
    *Qwen3*|*qwen3*)
        TC_PARSER=\"hermes\"
        # THINK_OFF=1: skip --reasoning-parser. Chat template already suppresses
        # thinking via enable_thinking=False; a server-side reasoning parser
        # (deepseek_r1) mis-routes the plain tool_call output to reasoning_content
        # when the chat template pre-emits <think></think>, breaking tool routing.
        if [ \"${THINK_OFF:-0}\" != \"1\" ]; then
            REASONING_ARGS=\"--enable-reasoning --reasoning-parser deepseek_r1\"
        fi
        ;;
esac
echo \"[container] tool-call-parser=\${TC_PARSER} reasoning=\${REASONING_ARGS:-(off)}\"
vllm serve ${LLAMA_CTR} \
    --host 127.0.0.1 --port 8000 \
    --served-model-name agent \
    --dtype float16 \
    --enforce-eager \
    --max-model-len ${VLLM_MAX_MODEL_LEN:-16384} \
    --gpu-memory-utilization ${VLLM_GPU_UTIL} \
    --enable-chunked-prefill \
    --max-num-batched-tokens ${VLLM_MAX_BATCHED:-8192} \
    --max-num-seqs ${VLLM_MAX_SEQS:-256} \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \${TC_PARSER} \
    \${REASONING_ARGS} \
    > /benchmarks/sweep_results/vllm_swebench_live_${MODEL_TAG}_${N_CORES}c_${TS}_server_agent.log 2>&1 &
AGENT_PID=\$!
trap 'kill \$AGENT_PID 2>/dev/null || true' EXIT

echo '[container] waiting for server (timeout 300s)…'
for i in \$(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo \"  port 8000 up after \${i}*5s\"
        break
    fi
    sleep 5
    [ \$i -eq 60 ] && { echo '  timeout'; tail -60 /benchmarks/sweep_results/vllm_swebench_live_${N_CORES}c_${TS}_server_agent.log; exit 1; }
done

echo '[container] launching bench…'
python3 /benchmarks/profiler_swebench_live/bench_swebench_live.py \
    /benchmarks/sweep_results/$(basename ${OUT_CSV}) ${NUM_TASKS}
" 2>&1 | tee -a "${LOG}"

echo "[$(date)] done; CSV: ${OUT_CSV}" | tee -a "${LOG}"
echo "[$(date)] rows: $(wc -l < ${OUT_CSV} 2>/dev/null || echo 0)" | tee -a "${LOG}"
echo "[$(date)] traces: $([ -f ${TRACES} ] && wc -l < ${TRACES} || echo 0)" | tee -a "${LOG}"
