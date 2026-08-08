#!/usr/bin/env bash
# ============================================================================
# Single entry point for ALL Thor data collection.
# Runs the three stages in order; each stage is independently resumable
# (a cell already present is skipped), so re-running continues where it stopped.
#
#   Stage 1  Master sweep (eager, num_runs=1: 3 warm-up + 1 measured, median
#            over generated tokens) → sweep_locked_thor.csv
#            Backs: latency/energy/MBU/roofline/quant/footprint figures.
#
#   Stage 2  Repeatability N=15 (eager, 15 fresh-load runs/cell)
#            → repeat_stats/*.jsonl (+ CV stats).
#
#   Stage 3  Throughput graphs-ON, N=15 (vLLM/SGLang GGUF at the throughput
#            cells 128/128 + 2048/2048; fp16 stays eager per the paper's
#            per-platform graph-capture convention — see ../../METHODOLOGY_NOTES.md;
#            Orin sm_87 runs these eager) → thor_cudagraph_15run_rows.csv.
#
#   Build    build_sweep_locked_thor_15run_raw.py folds the master grid, the
#            repeat_stats runs, AND the graphs-ON rows into ONE sweep
#            data/chat/sweep_locked_thor.csv (raw, ~15 rows/cell; plot loaders
#            average per cell). refine_pareto_cudagraph_15run.py updates the
#            separate Fig 11 pareto input. There is no separate cudagraph sweep.
#
# To reproduce elsewhere: set PROFILE_ROOT to the dir holding your Thor profile
# tree (data/benchmarks, data/models, work/) — everything else derives from it.
# Container image tags and clock-lock frequencies live at the top of each
# sub-script. See README.md.
# Usage:   bash run_thor_collection.sh [stage1|stage2|stage3|all]   (default: all)
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${1:-all}"
run() { echo; echo "######## $* ########  $(date)"; }

if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage1" ]; then
  run "STAGE 1/3 — master sweep (eager, num_runs=1)"
  bash "$HERE/sweep/run_thor_full_sweep.sh"
  bash "$HERE/sweep/run_thor_quant_grid.sh"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage2" ]; then
  run "STAGE 2/3 — repeatability N=15 (eager) + CV"
  N=15 bash "$HERE/repeat15/repeat_topup15.sh"
  python3 "$HERE/repeat15/agg_cv.py"       || true
  python3 "$HERE/repeat15/agg_cv_quant.py" || true
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage3" ]; then
  run "STAGE 3/3 — throughput graphs-ON N=15 (collection)"
  bash "$HERE/repeat15/sweep_thor_cudagraph_15run.sh" both
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "build" ]; then
  run "BUILD — fold master + repeat_stats + graphs-ON into ONE sweep_locked_thor.csv"
  python3 "$HERE/repeat15/build_sweep_locked_thor_15run_raw.py"
  python3 "$HERE/repeat15/refine_pareto_cudagraph_15run.py"
fi

run "COLLECTION COMPLETE"
