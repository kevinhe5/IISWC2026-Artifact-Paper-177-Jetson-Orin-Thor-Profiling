#!/usr/bin/env bash
# ============================================================================
# install_harness.sh — assemble the runtime benchmarks/ layout from the
# repo's deduplicated scripts/collect/harness/.
#
# The repo keeps ONE copy of each shared module (power_monitor.py,
# gpu_utils.py, memory_analysis.py, device_spec.py) and per-platform
# profiler dirs under harness/orin/. At runtime the containers mount
# ${PROFILE_ROOT}/benchmarks and set PYTHONPATH=/benchmarks/profiler_X
# (or nothing at all for llama.cpp), so every profiler dir must carry its
# own copy of the shared modules, the Orin profilers must sit at the top
# level, and device_spec.py must sit at benchmarks/ root (the benches
# reach it via a parent-dir sys.path insert).
#
# Called automatically by prepare_orin.sh; safe to re-run (pure overwrite
# from the repo; never deletes anything, sweep_results/ is untouched).
#
#   PROFILE_ROOT (or DATA_ROOT)  required — target is $PROFILE_ROOT/benchmarks
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS="${HERE}/../harness"
PROFILE_ROOT="${PROFILE_ROOT:-${DATA_ROOT:-$(cd "${HERE}/../../.." && pwd)/profile}}"
BENCH_DIR="${PROFILE_ROOT}/benchmarks"
mkdir -p "${BENCH_DIR}"

# Shared root module (parent-dir sys.path import contract)
cp -f "${HARNESS}/device_spec.py" "${BENCH_DIR}/"

# Platform-shared profiler dirs + Orin-specific ones flattened to top level
cp -rf "${HARNESS}/profiler_llamacpp"     "${BENCH_DIR}/"
cp -rf "${HARNESS}/profiler_trtllm"       "${BENCH_DIR}/"
cp -rf "${HARNESS}/orin/profiler_vllm"    "${BENCH_DIR}/"
cp -rf "${HARNESS}/orin/profiler_sglang"  "${BENCH_DIR}/"
cp -rf "${HARNESS}/orin/profiler_pytorch" "${BENCH_DIR}/"

# Shared helpers into every profiler dir (containers get at most one
# PYTHONPATH entry, so same-dir copies are the only portable option)
for d in profiler_llamacpp profiler_trtllm profiler_vllm profiler_sglang profiler_pytorch; do
    cp -f "${HARNESS}/power_monitor.py" "${BENCH_DIR}/${d}/"
    cp -f "${HARNESS}/gpu_utils.py"     "${BENCH_DIR}/${d}/"
done
cp -f "${HARNESS}/memory_analysis.py" "${BENCH_DIR}/profiler_pytorch/"

# Orin-specific pytorch gpu_utils overlay (extended helpers)
cp -f "${HARNESS}/harness_delta/orin/profiler_pytorch/gpu_utils.py" \
      "${BENCH_DIR}/profiler_pytorch/gpu_utils.py"

# ---- verify the import contract is satisfied ------------------------------
missing=0
need() { [ -f "${BENCH_DIR}/$1" ] || { echo "  MISSING: benchmarks/$1" >&2; missing=1; }; }
need device_spec.py
for d in profiler_llamacpp profiler_trtllm profiler_vllm profiler_sglang profiler_pytorch; do
    need "${d}/bench_e2e.py"
    need "${d}/power_monitor.py"
    need "${d}/gpu_utils.py"
done
need profiler_llamacpp/read_gguf.py
need profiler_pytorch/memory_analysis.py
if [ "$missing" -ne 0 ]; then
    echo "ERROR: harness install incomplete — see MISSING lines above." >&2
    exit 3
fi
echo "harness installed → ${BENCH_DIR} (5 profiler dirs, shared modules in place)"
