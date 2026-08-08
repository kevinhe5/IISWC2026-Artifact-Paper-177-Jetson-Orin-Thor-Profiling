#!/bin/bash
R=${PAPER_JETSON:-/nvme/ispass/paper_jetson}/repeat10
LOG=$R/orchestrator2_$(date +%Y%m%d_%H%M%S).log
echo "[$(date +%T)] waiting for outlier recheck to complete..." > $LOG
CUR=$(ls -t $R/outlier_*.log 2>/dev/null | head -1)
while true; do
    if [ -f "$CUR" ] && grep -q "OUTLIER RECHECK DONE" "$CUR" 2>/dev/null; then break; fi
    if [ -z "$(ps -ef | grep -iE 'run_outlier_recheck|bench_e2e' | grep -v grep)" ]; then break; fi
    sleep 60
done
echo "[$(date +%T)] outlier done, starting clean re-run" >> $LOG
bash $R/run_clean_rerun.sh 2>&1 | tee -a $LOG
echo "[$(date +%T)] ALL CLEAN RE-RUNS DONE" >> $LOG
