#!/usr/bin/env python3
"""Thor kernel-category extractor (Fig 6/7/8 artifact).

Recovers the wiped split_long_matmul_attention.py categorizer AND extends the
Orin extract_kernel_categories.py taxonomy so the Thor kernel names emitted by
each framework's CUDA impl land in the right architectural bucket:
  - flashinfer (SGLang) : BatchPrefill/Decode, MergeStates -> attention;
                          FusedAddRMSNorm/RMSNorm -> norm; BatchQKApplyRotary -> rope
  - Thor vLLM V1        : kernel_unified_attention_{2d,3d}, reduce_segments -> attention
  - TRT-Edge (Myelin)   : __myl_*Rope*/gemv_mha/kernel_mha/fmha/applyRope -> attention;
                          cutlass/gemvx/gemv/nvjet/__myl_*(fused GEMM) -> matmul;
                          __myl_*Mea*(RMSNorm) -> norm; __myl_*Silu* -> activation
  - llama.cpp           : k_set_rows -> copy_cast; unary_gated_op -> activation

Two front-ends produce the SAME per-fw schema as Orin kernel_categories.json
({categories:{cat:{ms_per_tok,ms_total,n_per_tok,fraction,top}}, total_ms,
gen_tokens, unclassified}):
  * categorize_sqlite(path, gen)  -- nsys CUPTI_ACTIVITY_KIND_KERNEL, grid-aware
        attention split for bare GEMM kernels (gridY>1 or gridZ>1 = multi-head bmm,
        which grows with KV length at long context; used by llama/vllm/pytorch/TRT).
  * categorize_chrome(path, gen)  -- torch-profiler chrome trace (name-based only;
        used for SGLang whose spawned worker escapes nsys fork-trace).
"""
import json, gzip, sqlite3, sys
from pathlib import Path

# First-match-wins. Superset of Orin extract_kernel_categories.RULES + Thor names.
RULES = [
    ("attention", [
        # generic / flash
        "kernel_mha", "flash_fwd", "flash_attn", "addBiasSoftMax",
        "soft_max", "softmax", "SoftMax", "fmha", "gemv_mha",
        # flashinfer (SGLang)
        "BatchPrefillWithPagedKVCache", "BatchDecodeWithPagedKVCache",
        "BatchPrefillWithRaggedKVCache", "VariableLengthMergeStates",
        "PersistentVariableLengthMergeStates", "MergeStates", "SinglePrefill",
        # vLLM V1 unified attention (Thor)
        "unified_attention", "reduce_segments", "paged_attention", "attention_kernel",
    ]),
    ("rope", [
        "rope_", "rotary_embedding", "applyBiasRope", "applyRopeWriteKV",
        "BatchQKApplyRotary", "rotary", "RopeCosSin",
    ]),
    ("kvcache", [
        "reshape_and_cache", "UpdateKVCache", "create_flashinfer_kv_indices",
    ]),
    ("norm", [
        "rms_norm", "fused_add_rms_norm", "FusedAddRMSNorm", "RMSNormKernel",
        "AddCasMulMeaAddSqrDivMulCasMul", "GatCasMulMeaAddSqrDivMulCasMul",
        "AddCasMulMea", "AddSqrDivMulCasMulAddGat",
        # TRT-Edge Myelin RMSNorm-like (Add/Sqrt/Div/Mean fused)
        "AddCastMulMeanAddSqrtDivMulCast", "CastMulMeanAddSqrtDivMulCast",
        "AddCastMulMean", "ReshCastAddSqrtDivMulCastMulGathMul",
    ]),
    ("activation", [
        "act_and_mul", "unary_op_kernel", "unary_gated_op",
        "CasCasMulTanMulAddMulMulCas", "silu", "gelu", "swiglu",
        "SiluMul",  # TRT-Edge Myelin SwiGLU (SiluMul / SiluMulMul)
    ]),
    ("quantize", [
        "quantize_q8_1", "quantize_mmq", "kDequantizeBlockwise",
        "dequantize_block", "kgemm_4bit",
    ]),
    ("matmul", [
        "mul_mat_vec_q", "mul_mat_vec", "mul_mat_q", "mul_mat",
        "kgemm_4bit_inference",
        "ampere_fp16_s16816gemm", "ampere_fp16_s1688", "s16816", "s1688",
        "cutlass_", "cutlass", "gemmSN_TN", "wmma_tensorop", "wmma", "tensorop",
        "_fuse_mul_mat", "gemmk1_kernel", "gemm",
        "dequantize_block_q6_K",
        "Kernel2",
        # Thor (sm_110) cuBLAS/cuBLASLt GEMM family + split-K reduce
        "nvjet", "splitKreduce", "splitkreduce",
        # TRT-Edge Myelin/cuBLAS weight matmul: gemvx GEMV, __myl_Fc fully-connected,
        # gemv weight GEMV, fused reshape-matmul.
        "gemvx", "gemv_kernel", "ReshMul", "_Fc_", "myl_Fc",
    ]),
    ("copy_cast", [
        "cpy_f32_f16", "cpy_f32_f32", "convert_unary",
        "CatArrayBatchedCopy", "k_get_rows_float", "k_get_rows", "k_set_rows",
        "indexSelectSmallIndex", "index_elementwise",
        "copyVectorizedKernel", "direct_copy", "copy_kernel", "Cast_",
        # TRT-Edge Myelin data movement: slice / reshape-scatter
        "Slic", "ReshScat", "ReshSpliMulMulMulMulSubAddConc", "ReshReshAddResh",
    ]),
    ("sampling", [
        "topK", "topk", "batchApplyPenalty",
        "curandInitialize", "lengthCriterion", "scatterDecodingParams",
        "copyNextStepIds", "incrementLengthTensor", "initializeNormalRope",
    ]),
    ("elementwise", [
        "k_bin_bcast", "k_bin", "vectorized_elementwise_kernel",
        "elementwise_kernel", "unrolled_elementwise_kernel", "reduce_kernel",
        "DeviceScanInitKernel", "DeviceScanKernel",
        "computeSeqAndPaddingOffsets", "triton_poi", "triton_",
    ]),
]

# GEMM name fragments used by the grid-aware attention split (lowercased match).
_GEMM = ("mul_mat", "gemm", "nvjet", "splitkreduce", "gemvx", "gemv", "cutlass",
         "wmma", "s16816", "s1688", "ampere_fp16")


def classify(name: str) -> str:
    for cat, pats in RULES:
        for pat in pats:
            if pat in name:
                return cat
    if name in ("kernel", "Kernel"):   # trtllm/pytorch bare cuBLAS name
        return "matmul"
    return "other"


def _grid_is_attention(name: str, gx, gy, gz) -> bool:
    """A GEMM-named kernel with gridY>1 (n_heads) and small gridX (head_dim tiles)
    is a multi-head attention batched-matmul (QK^T / AV), whose time grows with
    KV-cache length. The gridX<=256 guard keeps large weight GEMMs (full hidden-dim
    tiling) in matmul. Applied only to llama/pytorch/vllm/sglang bmm attention;
    NOT to TRT-Edge Myelin/cutlass GEMMs (those grid-split into false attention -
    per trt_long.sh, TRT attention is name-only via gemv_mha/fmha)."""
    n = name.lower()
    if "cutlass" in n or "tensorop" in n or "myl" in n:
        return False
    if not (any(g in n for g in _GEMM) or name in ("kernel", "Kernel", "kernel2", "Kernel2")):
        return False
    if not ((gy and gy > 1) or (gz and gz > 1)):
        return False
    return (gx or 0) <= 256


def _schema(cats, gen):
    """cats: {catname: {'ms':float,'n':int,'byname':{name:[ms,n]}}} -> schema dict."""
    out = {}
    total_ms = sum(v["ms"] for v in cats.values())
    unclass = []
    for cat, v in cats.items():
        kernels = [{"name": nm, "ms": mn[0], "n": int(mn[1])} for nm, mn in v["byname"].items()]
        kernels.sort(key=lambda x: -x["ms"])
        out[cat] = {
            "ms_per_tok": v["ms"] / gen,
            "ms_total": v["ms"],
            "n_per_tok": v["n"] / gen,
            "fraction": v["ms"] / total_ms if total_ms else 0,
            "top": kernels[:3],
        }
        if cat == "other":
            unclass = kernels
    return {"categories": out, "total_ms": total_ms, "gen_tokens": gen,
            "unclassified": unclass}


def _accumulate(rows, gen, grid_split):
    """rows: iterable of (name, ms, n, gx, gy, gz) ; gx/gy/gz may be None.
    Same kernel name may span multiple grid-groups -> merge by name within category."""
    cats = {}
    for name, ms, n, gx, gy, gz in rows:
        if ms <= 0:
            continue
        if grid_split and _grid_is_attention(name, gx, gy, gz):
            cat = "attention"
        else:
            cat = classify(name)
        c = cats.setdefault(cat, {"ms": 0.0, "n": 0, "byname": {}})
        c["ms"] += ms; c["n"] += int(n)
        bn = c["byname"].setdefault(name, [0.0, 0])
        bn[0] += ms; bn[1] += int(n)
    return _schema(cats, gen)


def categorize_sqlite(path, gen, grid_split=True):
    c = sqlite3.connect(str(path))
    rows = c.execute("""
        SELECT s.value, k.gridX, k.gridY, k.gridZ, SUM(k.end-k.start)/1e6 AS ms, COUNT(*) AS n
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
        GROUP BY s.value, k.gridX, k.gridY, k.gridZ
    """).fetchall()
    c.close()
    # collapse per (name) after grid decision, so 'top' lists whole kernels
    def gen_rows():
        for name, gx, gy, gz, ms, n in rows:
            yield (name or "", ms, n, gx, gy, gz)
    return _accumulate(gen_rows(), gen, grid_split)


def categorize_sqlite_named(path, gen):
    """Name-based only (no grid split) — matches the plain Orin extractor."""
    return categorize_sqlite(path, gen, grid_split=False)


def categorize_chrome(path, gen):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    ev = data.get("traceEvents", data) if isinstance(data, dict) else data
    agg = {}
    for e in ev:
        if not isinstance(e, dict):
            continue
        if "kernel" not in str(e.get("cat", "")).lower():
            continue
        dur = e.get("dur", 0); name = e.get("name", "")
        if not dur:
            continue
        a = agg.setdefault(name, [0.0, 0]); a[0] += dur / 1000.0; a[1] += 1  # us->ms
    rows = [(nm, ms, n, None, None, None) for nm, (ms, n) in agg.items()]
    return _accumulate(iter(rows), gen, grid_split=False)


if __name__ == "__main__":
    # CLI: extract_kernel_categories_thor.py OUT.json fw=SPEC [fw=SPEC ...]
    # SPEC = <kind>:<path>:<gen>[:namedonly]
    #   kind = sqlite | chrome
    out_path = sys.argv[1]
    result = {}
    for spec in sys.argv[2:]:
        fw, rest = spec.split("=", 1)
        parts = rest.split(":")
        kind, path, gen = parts[0], parts[1], int(parts[2])
        named = len(parts) > 3 and parts[3] == "namedonly"
        if kind == "sqlite":
            result[fw] = categorize_sqlite(path, gen, grid_split=not named)
        elif kind == "chrome":
            result[fw] = categorize_chrome(path, gen)
        else:
            print("bad kind", kind); sys.exit(2)
        cats = result[fw]["categories"]
        tot = result[fw]["total_ms"] / gen
        b = {k: 0.0 for k in ("matmul", "attention", "quantize", "copy_cast", "other")}
        for name, v in cats.items():
            b[name if name in b else "other"] += v["ms_per_tok"]
        print(f"{fw:12s} total={tot:6.2f} ms/tok | matmul={b['matmul']:.2f} attn={b['attention']:.2f} "
              f"quant={b['quantize']:.2f} copy={b['copy_cast']:.2f} other={b['other']:.2f}")
    Path(out_path).write_text(json.dumps(result, indent=2))
    print("wrote", out_path)
