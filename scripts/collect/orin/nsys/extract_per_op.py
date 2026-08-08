#!/usr/bin/env python3
"""Per-operation breakdown of the transformer decode step.

Uses the kernel timeline (sorted by start time) + grid-shape signatures to
label each kernel call as one of the 19 transformer operations:
  01_RMSNorm_in, 02_Q_proj, 03_K_proj, 04_V_proj, 05_RoPE,
  06_QKT_attention, 07_softmax, 08_AV_attention, 09_W_O,
  10_RMSNorm_post, 11_gate_proj, 12_up_proj, 13_SiLU, 14_mult_gate_up,
  15_down_proj, 16_residual_add, 17_RMSNorm_final, 18_KV_write, 19_lm_head

For each op we report median, p95, min, count — across all (16 layers × 128
tokens) instances.

Approach (llama.cpp / vLLM share the same MMVQ kernel pattern):
  - Every layer: 7 mul_mat_vec_q calls in order Q, K, V, O, Gate, Up, Down.
  - Disambiguate by gridX (2048: Q/O/Down, 512: K/V, 8192: Gate/Up,
    128256: lm_head)
  - Within a gridX class, position in sequence picks Q vs O vs Down etc.
  - mul_mat_vec (fp16 matvec) = attention matmuls (QK^T then AV).
"""
import json, sqlite3, statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent

# Llama-3.2-1B
N_LAYERS = 16
GEN_TOKENS = 128

# Per-framework kernel→op mapping logic.
# Each rule: (kernel_name_filter, op_label, position_within_class, gridX_match_or_None)

def label_llamacpp(events):
    """Label kernels by walking the timeline. Yields (op_label, duration_ns)."""
    # Per-layer state machine. State: layer_idx, position_within_layer.
    # We use kernel name + gridX to disambiguate.
    proj_seq = ['02_Q_proj', '03_K_proj', '04_V_proj', '09_W_O',
                '11_gate_proj', '12_up_proj', '15_down_proj']
    proj_grid = {2048: ['02_Q_proj', '09_W_O', '15_down_proj'],
                 512:  ['03_K_proj', '04_V_proj'],
                 8192: ['11_gate_proj', '12_up_proj']}

    # Track per-token state. A token = 7 layer-projections × 16 layers + 1 lm_head.
    # We rely on order of arrivals.
    out = []
    seen_per_grid = defaultdict(int)  # within current layer
    proj_idx_in_layer = 0
    layer_idx = 0
    attn_idx = 0  # 0=QKT, 1=AV
    in_layer_proj = None

    for name, gridX, gridY, dur in events:
        if name == 'mul_mat_vec_q':
            if gridX == 128256:
                out.append(('19_lm_head', dur))
                # lm_head signals end-of-token; reset state for next token.
                layer_idx = 0
                proj_idx_in_layer = 0
                seen_per_grid.clear()
                attn_idx = 0
                continue
            seq = proj_grid.get(gridX, [])
            if not seq:
                continue
            idx_in_class = seen_per_grid[gridX]
            if idx_in_class >= len(seq):
                # New layer started; reset.
                seen_per_grid.clear()
                idx_in_class = 0
                proj_idx_in_layer = 0
                layer_idx += 1
                attn_idx = 0
            label = seq[idx_in_class]
            seen_per_grid[gridX] += 1
            proj_idx_in_layer += 1
            out.append((label, dur))

            # When we've collected all 7 projections, advance layer.
            if proj_idx_in_layer >= 7:
                proj_idx_in_layer = 0
                seen_per_grid.clear()
                layer_idx += 1
                attn_idx = 0

        elif name == 'mul_mat_vec':
            # attention matmul (fp16): QK^T then AV
            label = '06_QKT_attention' if attn_idx == 0 else '08_AV_attention'
            attn_idx = 1 - attn_idx
            out.append((label, dur))

        elif name == 'rms_norm_f32':
            out.append(('01_RMSNorm', dur))
        elif name == 'rope_norm' or name.startswith('rope_'):
            out.append(('05_RoPE', dur))
        elif name == 'soft_max_f32':
            out.append(('07_softmax', dur))
        elif name == 'unary_op_kernel':
            out.append(('13_SiLU', dur))
        elif name == 'k_bin_bcast':
            # gridX=32 → gate × up element-wise multiply (8192-wide), 1/layer
            # gridX=8 → residual adds + small broadcasts (2048-wide), ~4/layer
            if gridX == 32:
                out.append(('14_mult_gate_up', dur))
            else:
                out.append(('16_residual_add', dur))
        elif name == 'cpy_f32_f16':
            out.append(('18_KV_write', dur))
        # else: ignore (cast/etc)
    return out


def label_vllm(events):
    """vLLM uses ggml's MMVQ kernel for projections (same gridX disambiguation),
    but its own dedicated kernels for the per-op transformer pieces."""
    proj_grid = {2048: ['02_Q_proj', '09_W_O', '15_down_proj'],
                 512:  ['03_K_proj', '04_V_proj'],
                 8192: ['11_gate_proj', '12_up_proj']}

    out = []
    seen_per_grid = defaultdict(int)
    proj_idx_in_layer = 0
    layer_idx = 0
    attn_idx = 0

    for name, gridX, gridY, dur in events:
        if name == 'mul_mat_vec_q' or name == 'mul_mat_q':
            if gridX == 128256:
                out.append(('19_lm_head', dur))
                layer_idx = 0; proj_idx_in_layer = 0
                seen_per_grid.clear(); attn_idx = 0
                continue
            seq = proj_grid.get(gridX, [])
            if not seq: continue
            idx_in_class = seen_per_grid[gridX]
            if idx_in_class >= len(seq):
                seen_per_grid.clear(); idx_in_class = 0
                proj_idx_in_layer = 0; layer_idx += 1; attn_idx = 0
            label = seq[idx_in_class]
            seen_per_grid[gridX] += 1
            proj_idx_in_layer += 1
            out.append((label, dur))
            if proj_idx_in_layer >= 7:
                proj_idx_in_layer = 0; seen_per_grid.clear()
                layer_idx += 1; attn_idx = 0

        elif name == 'fused_add_rms_norm_kernel':
            out.append(('01_RMSNorm', dur))
        elif name == 'rotary_embedding_kernel':
            out.append(('05_RoPE', dur))
        elif name == 'flash_fwd_splitkv_kernel' or name.startswith('flash_fwd'):
            # vLLM's PagedAttention/FlashInfer fuses QK^T + softmax + AV into one
            # kernel. We label as combined attention.
            out.append(('06-08_attention_fused', dur))
        elif name == 'cunn_SoftMaxForward':
            out.append(('07_softmax', dur))
        elif name == 'act_and_mul_kernel':
            # SwiGLU = SiLU(gate) * up — fused in vLLM
            out.append(('13-14_silu_mult', dur))
        elif name == 'reshape_and_cache_flash_kernel':
            out.append(('18_KV_write', dur))
        elif name in ('CatArrayBatchedCopy_alignedK_contig',):
            out.append(('16_residual_add', dur))
    return out


def aggregate(labeled):
    """Group by op label, compute median/p95/min/count."""
    by_op = defaultdict(list)
    for op, dur_ns in labeled:
        by_op[op].append(dur_ns)
    out = []
    for op, durs in sorted(by_op.items()):
        n = len(durs)
        durs_us = [d / 1000.0 for d in durs]  # ns → µs
        durs_sorted = sorted(durs_us)
        median = statistics.median(durs_us)
        p95 = durs_sorted[int(0.95 * n)] if n >= 20 else durs_sorted[-1]
        mn = durs_sorted[0]
        out.append({
            "op": op, "n": n,
            "median_us": median,
            "p95_us": p95,
            "min_us": mn,
            "p95_over_min": (p95 / mn) if mn > 0 else 0,
            "total_ms": sum(durs_us) / 1000.0,
        })
    return sorted(out, key=lambda x: -x["total_ms"])


def fetch_events(sql_path):
    conn = sqlite3.connect(str(sql_path))
    c = conn.cursor()
    rows = c.execute("""
SELECT s.value, k.gridX, k.gridY, (k.end - k.start) AS dur_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
ORDER BY k.start
""").fetchall()
    conn.close()
    return rows


def main():
    out = {}
    for fw, labeler in [('llamacpp', label_llamacpp), ('vllm', label_vllm)]:
        sql = ROOT / "traces" / f"{fw}_decode.sqlite"
        if not sql.exists():
            print(f"[skip] {fw}"); continue
        events = fetch_events(sql)
        labeled = labeler(events)
        agg = aggregate(labeled)
        out[fw] = agg
        print(f"\n=== {fw} per-op (over 128 tokens × ops/tok) ===")
        print(f"{'op':<22} {'n':>5} {'median µs':>10} {'p95 µs':>10} {'min µs':>9} {'p95/min':>8} {'total ms':>10}")
        for r in agg:
            print(f"{r['op']:<22} {r['n']:>5} {r['median_us']:>10.2f} {r['p95_us']:>10.2f} {r['min_us']:>9.2f} {r['p95_over_min']:>8.2f} {r['total_ms']:>10.2f}")
    (ROOT / "per_op.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT/'per_op.json'}")


if __name__ == "__main__":
    main()
