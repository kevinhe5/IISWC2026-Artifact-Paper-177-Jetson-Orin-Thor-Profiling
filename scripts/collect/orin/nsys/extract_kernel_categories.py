#!/usr/bin/env python3
"""Slice GPU kernel time by architectural category, per framework.

Uses substring rules tuned to the actual kernel names emitted by each
framework's CUDA implementation. Output is per-framework category breakdown
as JSON (consumed by /architecture page).

Categories (in priority order — first match wins):
  matmul       — weight-bound matmul (Q-mat, GEMM, kgemm_4bit)
  attention    — softmax + flash + multi-head attention kernels
  norm         — RMSNorm and friends
  rope         — rotary position embedding
  kvcache      — reshape_and_cache, applyBiasRopeUpdateKVCache
  quantize     — activation quantize/dequantize on the GPU (Q8_1, blockwise)
  activation   — silu/gelu/swiglu fused activation
  copy_cast    — f32↔f16 casts, tensor cat, etc.
  elementwise  — generic mul/add/broadcast kernels not covered above
  sampling     — topk, penalty, sample post-processing
  other        — anything not classified
"""
import json, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Each rule is (category, list of substring patterns).
# Order matters — first match wins. Tighten patterns where overlap is possible.
RULES = [
    ("attention", [
        "kernel_mha", "flash_fwd", "addBiasSoftMax",
        "soft_max", "softmax", "SoftMax",
        "fmha_v2_flash_attention",  # trtllm
    ]),
    ("rope", [
        "rope_", "rotary_embedding", "applyBiasRope",
    ]),
    ("kvcache", [
        "reshape_and_cache", "UpdateKVCache",
    ]),
    ("norm", [
        "rms_norm", "fused_add_rms_norm", "AddCasMulMeaAddSqrDivMulCasMul",
        "GatCasMulMeaAddSqrDivMulCasMul",
        # TRT-LLM Myelin-fused RMSNorm-like kernels — pattern is Add/Sqr/Div/Mea
        "AddCasMulMea", "AddSqrDivMulCasMulAddGat",
    ]),
    ("activation", [
        "act_and_mul", "unary_op_kernel", "CasCasMulTanMulAddMulMulCas",
    ]),
    ("quantize", [
        "quantize_q8_1", "quantize_mmq", "kDequantizeBlockwise",
    ]),
    ("matmul", [
        "mul_mat_vec_q", "mul_mat_vec", "mul_mat_q",
        "kgemm_4bit_inference",
        "ampere_fp16_s16816gemm", "ampere_fp16_s1688",
        "cutlass_", "gemmSN_TN", "wmma_tensorop",
        "_fuse_mul_mat", "gemmk1_kernel",
        # vLLM Q6_K embedding dequant counts as matmul-adjacent
        "dequantize_block_q6_K",
        # cuBLAS bare-name fallback in pytorch traces
        "Kernel2",
    ]),
    ("copy_cast", [
        "cpy_f32_f16", "cpy_f32_f32", "convert_unary",
        "CatArrayBatchedCopy", "k_get_rows_float",
        "indexSelectSmallIndex", "index_elementwise",
        "copyVectorizedKernel",
    ]),
    ("sampling", [
        "topK", "topk", "batchApplyPenalty",
        # trtllm sampling/decoding bookkeeping
        "curandInitialize", "lengthCriterion", "scatterDecodingParams",
        "copyNextStepIds",
    ]),
    # Generic broadcast / elementwise — must come AFTER more specific ones.
    ("elementwise", [
        "k_bin_bcast", "vectorized_elementwise_kernel",
        "elementwise_kernel", "unrolled_elementwise_kernel",
        "reduce_kernel",
        # TRT-LLM bookkeeping
        "DeviceScanInitKernel", "DeviceScanKernel",
        "computeSeqAndPaddingOffsets",
    ]),
]

# trtllm uses "kernel" as the W4A16 weight-only matmul name (no descriptive
# label). It's the only kernel that runs ~10000+ times on trtllm at decode.
# Catch it as a special-case matmul. Same for "Kernel" (capital).
TRTLLM_FALLBACK = [
    ("matmul", "^kernel$"),
    ("matmul", "^Kernel$"),
]

def classify(name: str) -> str:
    for cat, pats in RULES:
        for pat in pats:
            if pat in name:
                return cat
    # trtllm fallback (these are bare "kernel" / "Kernel" names)
    if name == "kernel" or name == "Kernel":
        return "matmul"
    return "other"


def categorize_trace(sql_path: Path, gen_tokens: int = 128) -> dict:
    conn = sqlite3.connect(str(sql_path))
    c = conn.cursor()

    # Detect NVTX prefill_only / full_gen ranges. If present, subtract the
    # prefill_only kernels from full_gen to recover pure decode (then scale
    # back to the gen_tokens basis).
    nvtx = {}
    try:
        for name, st, en in c.execute(
            "SELECT text, start, end FROM NVTX_EVENTS "
            "WHERE text IN ('prefill_only', 'full_gen')"
        ).fetchall():
            if not (st and en): continue
            if name not in nvtx or (en - st) > (nvtx[name][1] - nvtx[name][0]):
                nvtx[name] = (st, en)
    except sqlite3.OperationalError:
        pass
    has_nvtx = ('prefill_only' in nvtx and 'full_gen' in nvtx)

    if has_nvtx:
        f_st, f_en = nvtx['full_gen']
        p_st, p_en = nvtx['prefill_only']
        # Per-kernel-name: full window kernels MINUS prefill_only kernels,
        # then scaled to "as if 128 tokens".
        full = c.execute(f"""
SELECT s.value, SUM(k.end - k.start)/1e6 AS ms, COUNT(*) AS n
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
WHERE k.start >= {f_st} AND k.end <= {f_en}
GROUP BY s.value
""").fetchall()
        pref = c.execute(f"""
SELECT s.value, SUM(k.end - k.start)/1e6 AS ms, COUNT(*) AS n
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
WHERE k.start >= {p_st} AND k.end <= {p_en}
GROUP BY s.value
""").fetchall()
        pref_map = {r[0]: r for r in pref}
        scale = gen_tokens / max(gen_tokens - 1, 1)
        rows = []
        for name, fms, fn in full:
            pms = pref_map.get(name, (None, 0, 0))[1] or 0
            pn  = pref_map.get(name, (None, 0, 0))[2] or 0
            ms = max(fms - pms, 0) * scale
            n  = max(fn - pn, 0) * scale
            if ms > 0:
                rows.append((name, ms, int(n)))
    else:
        rows = c.execute("""
            SELECT s.value, SUM(k.end - k.start)/1e6 AS ms, COUNT(*) AS n
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
            GROUP BY s.value
        """).fetchall()
    conn.close()

    cats = {}
    unclass = []
    for name, ms, n in rows:
        cat = classify(name)
        cats.setdefault(cat, {"ms": 0.0, "n": 0, "kernels": []})
        cats[cat]["ms"] += ms
        cats[cat]["n"] += n
        cats[cat]["kernels"].append({"name": name, "ms": ms, "n": n})
        if cat == "other":
            unclass.append((name, ms, n))

    # Per-token
    out = {}
    total_ms = sum(v["ms"] for v in cats.values())
    for cat, v in cats.items():
        out[cat] = {
            "ms_per_tok": v["ms"] / gen_tokens,
            "ms_total": v["ms"],
            "n_per_tok": v["n"] / gen_tokens,
            "fraction": v["ms"] / total_ms if total_ms else 0,
            # top 3 kernels in this category by time
            "top": sorted(v["kernels"], key=lambda x: -x["ms"])[:3],
        }
    return {"categories": out, "total_ms": total_ms, "gen_tokens": gen_tokens,
            "unclassified": [{"name": n, "ms": m, "n": k} for n, m, k in unclass]}


def main():
    out = {}
    for fw in ["trtllm", "llamacpp", "vllm", "pytorch"]:
        sql = ROOT / "traces" / f"{fw}_decode.sqlite"
        if not sql.exists():
            print(f"[skip] {fw}")
            continue
        out[fw] = categorize_trace(sql)
        cats = out[fw]["categories"]
        print(f"\n=== {fw} (total GPU kernel = {out[fw]['total_ms']/128:.2f} ms/tok) ===")
        order = ["matmul", "attention", "norm", "rope", "kvcache", "quantize",
                 "activation", "copy_cast", "elementwise", "sampling", "other"]
        for c in order:
            if c not in cats: continue
            v = cats[c]
            print(f"  {c:<12} {v['ms_per_tok']:6.3f} ms/tok  ({v['fraction']*100:5.1f}%)  "
                  f"n={v['n_per_tok']:5.1f}/tok")
        if out[fw]["unclassified"]:
            print(f"  unclassified: {len(out[fw]['unclassified'])} kernel name(s):")
            for u in out[fw]["unclassified"][:5]:
                print(f"    {u['ms']:6.2f} ms  {u['name'][:80]}")

    (ROOT / "kernel_categories.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT/'kernel_categories.json'}")


if __name__ == "__main__":
    main()
