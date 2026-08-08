#!/bin/bash
# Orchestrator: Fig 3 → Fig 5 only.
# (Contention removed — Table 3 is Thor data, not applicable on Orin.)
# (Fig 2 removed per user request.)

R=${PAPER_JETSON:-/nvme/ispass/paper_jetson}/repeat10
LOG=$R/orchestrator_$(date +%Y%m%d_%H%M%S).log
touch "$LOG"
log() { echo "[$(date +'%T')] $*" | tee -a "$LOG"; }

log "orchestrator started"

log "→ starting FIG 3 (pp=512 gen=256)"
bash $R/run_fig3_mbu.sh 2>&1 | tee -a "$LOG"
log "  FIG 3 done"

log "→ starting FIG 5 gap-fill (I ∈ {256, 512, 1024, 2048})"
bash $R/run_fig5_gapfill.sh 2>&1 | tee -a "$LOG"
log "  FIG 5 done"

log "ORCHESTRATOR DONE"
