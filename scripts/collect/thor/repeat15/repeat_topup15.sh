#!/bin/bash
# Top-up EVERY repeatability cell to N=15 fully-independent (fresh-load) runs, paper-consistent runners.
# Runners now append the deficit (15 - existing) rather than truncate, so prior reps are kept.
# Fully resumable: re-running continues wherever it stopped. Detached (survives session end).
set -u
export N=15
S=${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/thor/scripts
echo "########## TOPUP N=15 START $(date) ##########"
echo "== [1/4] fp16 5-fw 6x6 grid =="            ; bash $S/repeat_grid.sh
echo "== [2/4] quant/variant grids =="           ; bash $S/repeat_grid_quant.sh
echo "== [3/4] trtedge fp16 axis =="             ; bash $S/trtedge_repeat.sh
echo "== [4/4] trtedge quant 6x6 grids =="       ; bash $S/repeat_grid_trtedge_quant.sh
echo "########## TOPUP RUNNERS DONE $(date) — final CV agg ##########"
python3 $S/agentic/agg_cv.py 2>&1
python3 $S/agentic/agg_cv_quant.py 2>&1
echo "########## ALL DONE $(date) ##########"
