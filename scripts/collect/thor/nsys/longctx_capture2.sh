#!/bin/bash
# Fig8 long-ctx (round 2): vllm + sglang + pytorch at gen=65536 with warmup=0
# (single full pass -> no timeout truncation; kernel-mix needs no warmup, clocks locked).
# llamacpp already captured+extracted in round 1. pytorch runs LAST (slowest, unattended).
set -u
RG="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work"
T="${NSYS_TRACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/data/nsys/traces}"
G=65536
LOG="$RG/logs/longctx_master2.log"
echo "######## FIG8 LONG-CTX R2 gen=$G warmup=0 $(date) ########" > "$LOG"

do_fw(){
  local fw="$1" tag="${1}_decode_g${G}"
  echo "#### $fw gen=$G START $(date) ####" | tee -a "$LOG"
  "$RG/nsys_capture.sh" "$fw" "$G" "$tag" 0 >>"$LOG" 2>&1
  if [ -f "$T/$tag.sqlite" ]; then
    python3 - "$T/$tag.sqlite" "$fw" "$G" >>"$LOG" 2>&1 <<'PY'
import sys; sys.path.insert(0,"${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work")
from extract_kernel_categories_thor import categorize_sqlite
import json
sql,fw,gen=sys.argv[1],sys.argv[2],int(sys.argv[3])
try:
    r=categorize_sqlite(sql,gen,grid_split=True)
    open("${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work/logs/longctx_g65536.jsonl","a").write(json.dumps({fw:r})+"\n")
    cats=r["categories"]; b={k:0.0 for k in ("matmul","attention","quantize","copy_cast","other")}
    for n,v in cats.items(): b[n if n in b else "other"]+=v["ms_per_tok"]
    print(f"  EXTRACT {fw}: total={r['total_ms']/gen:.2f} matmul={b['matmul']:.2f} attn={b['attention']:.2f} other={b['other']:.2f}")
except Exception as e:
    print(f"  EXTRACT-FAIL {fw}: {e}")
PY
  else
    echo "  NO-SQLITE $fw (capture failed / no kernels)" | tee -a "$LOG"
  fi
  echo "#### $fw gen=$G DONE $(date) ####" | tee -a "$LOG"
  grep -E "EXTRACT|NO-SQLITE|rep=" "$LOG" | tail -3
}

for fw in vllm sglang pytorch; do do_fw "$fw"; done
echo "######## FIG8 LONG-CTX R2 DONE $(date) ########" | tee -a "$LOG"
