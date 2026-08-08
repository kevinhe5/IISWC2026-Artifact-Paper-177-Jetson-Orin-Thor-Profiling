#!/bin/bash
# Fig8 long-context: capture decode nsys at pp=128, gen=65536 for each framework,
# sequentially (one GPU capture at a time, clean-run discipline via nsys_capture.sh).
# Then extract kernel categories per fw into a jsonl (partial progress preserved).
# ~50-65 min per fw. llamacpp/pytorch/vllm use nsys (single-process, works);
# sglang worker is spawned and escapes nsys fork-trace -> attempted, may yield no kernels.
set -u
RG="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work"
T="${NSYS_TRACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/data/nsys/traces}"
G=65536
LOG="$RG/logs/longctx_master.log"
echo "######## FIG8 LONG-CTX gen=$G $(date) ########" > "$LOG"

do_fw(){ # fw
  local fw="$1" tag="${1}_decode_g${G}"
  echo "#### $fw gen=$G START $(date) ####" | tee -a "$LOG"
  "$RG/nsys_capture.sh" "$fw" "$G" "$tag" 1 >>"$LOG" 2>&1
  if [ -f "$T/$tag.sqlite" ]; then
    python3 - "$T/$tag.sqlite" "$fw" "$G" >>"$LOG" 2>&1 <<'PY'
import sys; sys.path.insert(0,"${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work")
from extract_kernel_categories_thor import categorize_sqlite
import json
sql,fw,gen=sys.argv[1],sys.argv[2],int(sys.argv[3])
r=categorize_sqlite(sql,gen,grid_split=True)
open("${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/work/logs/longctx_g65536.jsonl","a").write(json.dumps({fw:r})+"\n")
cats=r["categories"]; b={k:0.0 for k in ("matmul","attention","quantize","copy_cast","other")}
for n,v in cats.items(): b[n if n in b else "other"]+=v["ms_per_tok"]
print(f"  EXTRACT {fw}: total={r['total_ms']/gen:.2f} matmul={b['matmul']:.2f} attn={b['attention']:.2f} other={b['other']:.2f}")
PY
  fi
  echo "#### $fw gen=$G DONE $(date) ####" | tee -a "$LOG"
  grep -E "EXTRACT|exit=|rep=" "$LOG" | tail -3
}

: > "$RG/logs/longctx_g65536.jsonl"
for fw in llamacpp pytorch vllm sglang; do do_fw "$fw"; done
echo "######## FIG8 LONG-CTX DONE $(date) ########" | tee -a "$LOG"
