#!/bin/bash
# Run nsys profile 3× for the same framework, collecting tpot from each run
# to estimate nsys-overhead variance (Phase 2b — issue #10).
# Usage: ./run_profile_repeat.sh {trtllm|llamacpp|vllm|pytorch} [n_repeats]
set -e

FW="$1"
N_REPEATS="${2:-3}"
DATA_ROOT="${DATA_ROOT:-${PROFILE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/profile}}"
ROOT="${DATA_ROOT}/benchmarks/nsys_profiles"
LOGDIR="${ROOT}/logs/repeat_${FW}"
mkdir -p "${LOGDIR}"

OUT="${ROOT}/nsys_overhead_${FW}.json"
TMP="${LOGDIR}/runs.txt"
> "${TMP}"

echo "Running ${N_REPEATS} profiled runs of ${FW}..."
for i in $(seq 1 "${N_REPEATS}"); do
    echo "--- run ${i}/${N_REPEATS} ---"
    log="${LOGDIR}/run_${i}.log"
    "${ROOT}/run_profile.sh" "${FW}" 2>&1 | tee "${log}" | tail -5
    # Parse the [framework] median tpot line
    tpot=$(grep -oE "median tpot=[0-9.]+" "${log}" | tail -1 | sed 's/median tpot=//')
    echo "${i} ${tpot}" >> "${TMP}"
done

# Compute median + std of the median-tpots
python3 - <<EOF
import json, statistics
with open("${TMP}") as f:
    vals = [float(line.split()[1]) for line in f if line.strip()]
out = {
    "framework": "${FW}",
    "n_profiled_runs": len(vals),
    "median_tpots_ms": vals,
    "median_of_medians_ms": statistics.median(vals),
    "stdev_of_medians_ms": statistics.stdev(vals) if len(vals) > 1 else 0,
    "max_minus_min_ms": max(vals) - min(vals),
}
with open("${OUT}", "w") as f:
    json.dump(out, f, indent=2)
print(f"\\n{out['framework']:<10} n={out['n_profiled_runs']}  "
      f"median={out['median_of_medians_ms']:.3f} ms  "
      f"std={out['stdev_of_medians_ms']:.3f} ms  "
      f"range={out['max_minus_min_ms']:.3f} ms")
EOF
