#!/usr/bin/env bash
# ============================================================================
# Single entry point for ALL Orin data collection.
# Mirrors scripts/collect/thor/run_thor_collection.sh (same envvar, same
# stage-flag interface: stage1 | side | stage2 | stage3 | build | all).
# This one script produces every Orin-side file the plotting scripts read.
#
# Each stage is independently resumable (a cell already present is skipped),
# so re-running continues where it stopped.
#
#   Stage 1  Master sweep (sweep/sweep.sh, SWEEP_SCOPE=full: 5 fw incl.
#            SGLang × {1B, 8B, Mixtral} × quants; eager, num_runs=1:
#            3 warm-up + 1 measured) → sweep_results/sweep_locked_master.csv
#            Backs: latency/energy/MBU/roofline/quant/footprint figures.
#            Wall-clock: ~7 h on a 32 GB AGX Orin at locked clocks.
#
#   Side     Three small extra grids (side_sweeps.sh) that back the
#            remaining paper series:
#              compile  torch.compile 8 cells   → data/chat/pytorch_compile.csv
#              fa       llama.cpp FA-on 16 cells → data/chat/llamacpp_fa_orin.csv
#              longctx  long decode 27 cells     → data/chat/longctx_fp16_orin.csv
#            Wall-clock: ~1 h (compile+fa) + ~8-10 h (longctx; 131K-token
#            decodes). Subset with SIDE=compile|fa|longctx.
#
#   Stage 2  Repeatability N=REP_COUNT (default 15, matches AE claim
#            "N=15 means match a single-run sweep within 0.6%") — the same
#            worker re-run with SWEEP_SCOPE=1b (skips 8B/Mixtral), one
#            fresh container-loaded pass per rep to characterize
#            cross-execution CV. Override with REP_COUNT=3 for the ~21 h
#            quick pass or REP_COUNT=15 for the full ~4.4 day dataset.
#            → sweep_results/rep15/rep{1..N}.csv
#
#   Stage 3  Nsight Systems kernel decomposition. On sm_87 there is no
#            graphs-ON path (Thor's Stage 3 is Thor-only per
#            METHODOLOGY_NOTES.md); Orin's Stage 3 instead re-captures the
#            per-token (kernel / launch / residual) breakdown that backs
#            Fig 4/9/10. Wall-clock: ~2 h.
#            → data/nsys/{breakdown.json, kernel_categories.json,
#                         per_op.json, all_overhead_summary.json}
#
#   Build    build_sweep_locked_orin.py folds the master sweep + rep15
#            runs into ONE data/chat/sweep_locked.csv (raw, N+1 rows/cell;
#            plot loaders average per cell). Also regenerates
#            data/chat/mbu_pp512_gen256.csv Orin rows (Thor rows preserved
#            from shipped) and projects the side grids into
#            pytorch_compile.csv, llamacpp_fa_orin.csv, longctx_fp16_orin.csv.
#
# Environment (per AE §5-7 contract):
#   PROFILE_ROOT  required — directory holding data/benchmarks/,
#                            data/models/ and a work/ scratch subdir.
#                            Aliased internally to DATA_ROOT for the
#                            existing sub-scripts.
#   REPO_ROOT     auto-detected (parent of scripts/collect/orin/)
#   REP_COUNT     15   (or 3 for a quicker cross-execution check)
#   NSYS          1    (0 skips Stage 3)
#   SIDE          all  (or compile|fa|longctx to subset the side stage)
#
# Usage:  bash run_orin_collection.sh [stage1|side|stage2|stage3|build|all]
#         default: all
#
# Estimated total wall-clock at defaults (REP_COUNT=15):
#   Stage 1 ~7 h  ·  Side ~10 h  ·  Stage 2 ~105 h  ·  Stage 3 ~2 h  ·  Build ~5 min
# For a reviewer time-box (~ 8 h), REP_COUNT=1 gives the minimum meaningful
# cross-execution replicate; REP_COUNT=3 (~ 28 h total) matches the anchor
# 3-rep dataset used by the paper's §B.1.1 repeatability envelope.
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${HERE}/../../.." && pwd)}"

# PROFILE_ROOT is the AE-facing envvar (legacy alias: DATA_ROOT). Defaults
# to <repo>/profile/ (gitignored); export PROFILE_ROOT to place the ~60 GB
# of models/outputs on a different filesystem.
PROFILE_ROOT="${PROFILE_ROOT:-${DATA_ROOT:-${REPO_ROOT}/profile}}"
echo "PROFILE_ROOT = ${PROFILE_ROOT}"
export DATA_ROOT="${PROFILE_ROOT}"   # alias for existing sub-scripts

REP_COUNT="${REP_COUNT:-15}"
NSYS="${NSYS:-1}"
STAGE="${1:-all}"
case "$STAGE" in
    all|stage1|side|stage2|stage3|build) ;;
    *) echo "ERROR: unknown stage '$STAGE' (expected stage1|side|stage2|stage3|build|all)" >&2; exit 1 ;;
esac

export PROFILE_ROOT DATA_ROOT REPO_ROOT

BENCH_DIR="${PROFILE_ROOT}/benchmarks"
SWEEP_OUT_DIR="${BENCH_DIR}/sweep_results"

run(){ echo; echo "######## $* ######## $(date -u +%FT%TZ)"; }
die(){ local rc=$?; echo; echo "FATAL: $* (exit ${rc})" >&2
       echo "       Fix the cause and re-run — completed cells are skipped on resume." >&2
       exit 1; }

# ---- Preflight (also invoked implicitly by sweep.sh) ---------------------
# The build stage is pure CSV folding — no GPU, no clocks, no sudo — so it
# must not be blocked by the hardware preflight (which needs passwordless
# sudo for jetson_clocks / swappiness / drop_caches).
if [ "$STAGE" = "build" ]; then
    run "PREFLIGHT (skipped — build stage does no measurement)"
else
    run "PREFLIGHT"
    bash "${HERE}/preflight.sh" || die "preflight failed"
fi

# ==========================================================================
# Stage 1 — master sweep
# ==========================================================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage1" ]; then
    run "STAGE 1/3 — master sweep (Llama-3.2-1B, 5 fw, eager, num_runs=1)"
    mkdir -p "${SWEEP_OUT_DIR}"
    export OUTPUT_CSV="${SWEEP_OUT_DIR}/sweep_locked_master.csv"
    cd "${BENCH_DIR}"
    SWEEP_SCOPE=full bash "${HERE}/sweep/sweep.sh" || die "stage1 master sweep failed"
fi

# ==========================================================================
# Side — compile / flash-attn / long-context grids
# ==========================================================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "side" ]; then
    run "SIDE — compile/FA/longctx grids (SIDE=${SIDE:-all})"
    cd "${BENCH_DIR}"
    SIDE="${SIDE:-all}" bash "${HERE}/side_sweeps.sh" || die "side sweeps failed"
fi

# ==========================================================================
# Stage 2 — repeatability N=REP_COUNT
# ==========================================================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage2" ]; then
    run "STAGE 2/3 — repeatability N=${REP_COUNT}"
    mkdir -p "${SWEEP_OUT_DIR}/rep15"
    for rep in $(seq 1 "$REP_COUNT"); do
        csv="${SWEEP_OUT_DIR}/rep15/rep${rep}.csv"
        run "  rep ${rep}/${REP_COUNT}  →  ${csv}"
        bash "${HERE}/preflight.sh" || die "preflight failed before rep ${rep}"
        export OUTPUT_CSV="${csv}"
        cd "${BENCH_DIR}"
        SWEEP_SCOPE=1b bash "${HERE}/sweep/sweep.sh" || die "stage2 rep ${rep} failed"
    done
fi

# ==========================================================================
# Stage 3 — nsys kernel decomposition
# ==========================================================================
if { [ "$STAGE" = "all" ] || [ "$STAGE" = "stage3" ]; } && [ "$NSYS" = "1" ]; then
    run "STAGE 3/3 — nsys kernel decomposition (Fig 4/9/10 data)"
    bash "${HERE}/nsys/run_baseline.sh"       || die "nsys baseline capture failed"
    bash "${HERE}/nsys/run_profile.sh"        || die "nsys profile capture failed"
    bash "${HERE}/nsys/run_profile_repeat.sh" || die "nsys repeat capture failed"
    python3 "${HERE}/nsys/extract_breakdown.py"         > "${REPO_ROOT}/data/nsys/breakdown.json"            || die "extract_breakdown failed"
    python3 "${HERE}/nsys/extract_kernel_categories.py" > "${REPO_ROOT}/data/nsys/kernel_categories.json"    || die "extract_kernel_categories failed"
    python3 "${HERE}/nsys/extract_per_op.py"            > "${REPO_ROOT}/data/nsys/per_op.json"               || die "extract_per_op failed"
    python3 "${HERE}/nsys/launch_gap_dist.py"           > "${REPO_ROOT}/data/nsys/all_overhead_summary.json" || die "launch_gap_dist failed"
fi

# ==========================================================================
# Build — fold raw sweep + rep15 → data/chat/sweep_locked.csv
# ==========================================================================
if [ "$STAGE" = "all" ] || [ "$STAGE" = "build" ]; then
    run "BUILD — fold raw sweeps into the data/chat/ files the plots read"
    python3 "${HERE}/build_sweep_locked_orin.py" \
        --master "${SWEEP_OUT_DIR}/sweep_locked_master.csv" \
        --rep15-dir "${SWEEP_OUT_DIR}/rep15" \
        --side-dir  "${SWEEP_OUT_DIR}" \
        --out         "${REPO_ROOT}/data/chat/sweep_locked.csv" \
        --mbu-out     "${REPO_ROOT}/data/chat/mbu_pp512_gen256.csv" \
        --compile-out "${REPO_ROOT}/data/chat/pytorch_compile.csv" \
        --fa-out      "${REPO_ROOT}/data/chat/llamacpp_fa_orin.csv" \
        --longctx-out "${REPO_ROOT}/data/chat/longctx_fp16_orin.csv" \
        || die "build step failed"
fi

run "COLLECTION COMPLETE"
echo
echo "Next: re-render the paper figures against the regenerated data:"
echo "  for s in ${REPO_ROOT}/scripts/plot/gen_fig*.py; do python3 \"\$s\" --out figs/; done"
echo
echo "Agentic (§VI SWE-bench-live, Fig 12) is NOT part of the main collection"
echo "pipeline — matches Thor. Run the per-framework drivers separately:"
echo "  bash ${HERE}/agentic/rerun_swebench_live_{vllm,sglang,trtllm,llamacpp,pytorch}.sh"
