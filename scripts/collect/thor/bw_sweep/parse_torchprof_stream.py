#!/usr/bin/env python3
# Streaming parser for LARGE torch-profiler chrome traces (.json.gz) — no full json.load (avoids OOM
# on multi-GB traces). Splits the traceEvents array on '},{' boundaries in streamed chunks, json.loads
# each small event. Name-based attention (BatchPrefill etc.). Usage: <trace.gz> <gen> <tag>
import sys, gzip, json, re
_SEP = re.compile(r'\}\s*,\s*\{')
path, gen, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ATTN=("batchprefill","batchdecode","paged_kv","mergestate","variablelengthmerge","soft_max","softmax",
      "flash","unified_attention","reduce_segments","paged_attention","attention_kernel","fmha","flashinfer","fattn")
GEMM=("nvjet","gemm","splitkreduce","gemv","cutlass","wmma","mul_mat")
COPY=("cpy_","convert_unary","catarray","k_get_rows","indexselect","index_elementwise","copyvectorized","direct_copy")
QUANT=("quantize","dequant","kdequant")
NORM=("rms_norm","fused_add_rms_norm"); ROPE=("rope","rotary"); ACT=("act_and_mul","silu","gelu","unary_gated")
ELEM=("bin_bcast","vectorized_elementwise","elementwise_kernel","unrolled_elementwise","reduce_kernel","k_bin")
cat={k:0.0 for k in ("matmul","attention","copy_cast","quantize","norm","rope","activation","elementwise","other")}
def classify(n,d):
    n=n.lower()
    if any(a in n for a in ATTN): cat["attention"]+=d
    elif any(g in n for g in GEMM): cat["matmul"]+=d
    elif any(x in n for x in COPY): cat["copy_cast"]+=d
    elif any(x in n for x in QUANT): cat["quantize"]+=d
    elif any(x in n for x in NORM): cat["norm"]+=d
    elif any(x in n for x in ROPE): cat["rope"]+=d
    elif any(x in n for x in ACT): cat["activation"]+=d
    elif any(x in n for x in ELEM): cat["elementwise"]+=d
    else: cat["other"]+=d
def handle(obj):
    try: e=json.loads(obj)
    except Exception: return
    if "kernel" not in str(e.get("cat","")).lower(): return
    d=e.get("dur",0)
    if d: classify(e.get("name",""), d)
buf=""; started=False; n_ev=0
with gzip.open(path,"rt") as f:
    while True:
        chunk=f.read(8<<20)  # 8MB
        if not chunk: break
        buf+=chunk
        if not started:
            i=buf.find('"traceEvents"')
            if i<0:
                buf=buf[-32:]; continue
            j=buf.find('[', i)
            if j<0: continue
            buf=buf[j+1:]; started=True
        parts=_SEP.split(buf)
        buf=parts.pop()  # last is incomplete
        for p in parts:
            s=p if p.lstrip().startswith('{') else '{'+p
            s=s if s.rstrip().endswith('}') else s+'}'
            handle(s); n_ev+=1
# final remainder (strip trailing array close)
tail=buf.rstrip().rstrip(']').rstrip().rstrip(',')
if tail:
    s=tail if tail.lstrip().startswith('{') else '{'+tail
    if not s.rstrip().endswith('}'): s+='}'
    handle(s); n_ev+=1
tot=sum(cat.values())
print(f"  {tag}(÷{gen}, {n_ev} events): total/tok={tot/1000/gen:.3f} matmul={cat['matmul']/1000/gen:.3f} attn={cat['attention']/1000/gen:.3f} quant={cat['quantize']/1000/gen:.3f} other={cat['other']/1000/gen:.3f}")
