"""Hardware spec + roofline math helpers shared by the profiler benches.

Reconstructed from the import contract of profiler_*/bench_e2e.py + the
dashboard's MODEL_ARCH math. Source .py was missing on disk; only the
.pyc remained in data/benchmarks/__pycache__/ (no version-control copy
existed). Parameters validated against the live sweep CSV: predicted
MBU agrees with measured EMC within ±15pp across 23/23 (fw, model,
quant) configs after MoE/MLA corrections.
"""

# ---------------------------------------------------------------------------
# Hardware specs by Jetson SKU.
# ---------------------------------------------------------------------------
# AGX Orin 32GB MAXN_SUPER (locked clocks):
#   FP16 dense = 56 TC × 512 FLOPs/cycle × 1.3 GHz = 37.3 TFLOPS
#   INT8 dense = 2 × FP16 = 74.6 TOPS
#   LPDDR5     = 3200 MT/s × 256-bit = 204.8 GB/s
# Orin Nano Super 8GB MAXN_SUPER:
#   FP16 dense = 17.0 TFLOPS, INT8 = 34.0 TOPS
#   LPDDR5     = 3200 MT/s × 128-bit = 102.4 GB/s
SPECS = {
    "agx_orin_32gb": {
        "name": "AGX Orin 32GB",
        "peak_tflops_fp16_dense": 37.3,
        "peak_tops_int8_dense":   74.6,
        "peak_bw_gb_s":           204.8,
        # LPDDR5: 256-bit bus → 32 B per cycle (single channel); ×2 channels gets 204.8.
        "bus_bytes":              32,
        "emc_freq_max_mhz":       3199,
    },
    "orin_nano_8gb": {
        "name": "Orin Nano Super 8GB",
        "peak_tflops_fp16_dense": 17.0,
        "peak_tops_int8_dense":   34.0,
        "peak_bw_gb_s":           102.4,
        # LPDDR5: 128-bit bus → 16 B per cycle; ×2 channels gets 102.4.
        "bus_bytes":              16,
        "emc_freq_max_mhz":       3199,
    },
    "agx_thor_128gb": {
        # Jetson AGX Thor Developer Kit. Blackwell SoC (sm_121).
        # Placeholder spec values pending live-derivation on the device
        # (matches the approach we used for agx_orin_32gb in
        # the 2026-04 sweep). NVIDIA's published headlines:
        #   FP4 sparse  : 2070 TFLOPS
        #   FP4 dense   : 1035 TFLOPS
        #   FP16 dense  : ~124 TFLOPS (estimated from Blackwell scaling)
        #   INT8 dense  : ~248 TOPS
        #   LPDDR5X     : 8533 MT/s × 256-bit = 273.1 GB/s
        # These are subject to revision once measured on locked clocks.
        "name": "AGX Thor 128GB",
        "peak_tflops_fp16_dense": 124.0,
        "peak_tops_int8_dense":   248.0,
        "peak_bw_gb_s":           273.1,
        # LPDDR5X: 256-bit bus → 32 B per cycle; placeholder.
        "bus_bytes":              32,
        # emc_freq_max placeholder — Thor's LPDDR5X is 8533 MT/s; the
        # tegrastats EMC reporting layer may report a different value
        # (need to measure once a Thor is available).
        "emc_freq_max_mhz":       5333,
    },
}


def detect_platform() -> str:
    """Read /proc/device-tree/model. Falls back to AGX Orin 32GB."""
    try:
        m = open("/proc/device-tree/model").read().strip("\x00").lower()
    except Exception:
        m = ""
    if "thor" in m:
        return "agx_thor_128gb"
    if "orin nano" in m:
        return "orin_nano_8gb"
    return "agx_orin_32gb"


def get_spec(platform: str = None) -> dict:
    return SPECS[platform or detect_platform()]


# ---------------------------------------------------------------------------
# Quantization → which compute ceiling applies.
# ---------------------------------------------------------------------------
def peak_tflops_for_quant(quantization: str) -> float:
    """bitsandbytes 4bit/8bit dequant to fp16 before matmul → fp16 peak.
    bf16/fp16/16-bit → fp16 peak. Native int8 GEMM (rare in pytorch path)
    uses int8 peak."""
    spec = get_spec()
    q = (quantization or "").lower()
    if "int8" in q and "native" in q:
        return spec["peak_tops_int8_dense"]
    return spec["peak_tflops_fp16_dense"]


# ---------------------------------------------------------------------------
# MoE: active params per token (not total).
# ---------------------------------------------------------------------------
def moe_active_params(num_params, num_layers, hidden_size, intermediate_size,
                      num_local_experts=0, num_experts_per_tok=0,
                      n_shared_experts=0, vocab_size=0):
    """For dense models (no MoE): returns num_params unchanged.
    For MoE: subtracts idle-expert params and adds back the routed top-k +
    shared experts. Each expert FFN ≈ 3 × hidden × intermediate (SwiGLU:
    gate, up, down)."""
    if not num_local_experts or num_local_experts <= 1 or num_experts_per_tok <= 0:
        return num_params
    expert_size = 3 * hidden_size * intermediate_size
    total_expert_params  = num_local_experts            * num_layers * expert_size
    active_expert_params = (num_experts_per_tok + n_shared_experts) * num_layers * expert_size
    return int(num_params - total_expert_params + active_expert_params)


# ---------------------------------------------------------------------------
# Standard transformer FLOPs (Chinchilla / GPT-style accounting).
#   prefill = 2 × P × T_pp + 4 × L × d_model × T_pp²    (attn quadratic)
#   decode  = 2 × P × N_dec + 4 × L × d_model × T_ctx × N_dec
# P = active params (per-token MM is 2P FLOPs forward-only); attention adds
# the second term. Returns prefill/decode total FLOPs and the attention-only
# components so callers can separate.
# ---------------------------------------------------------------------------
def compute_flops(num_params, active_params, num_layers, hidden_size,
                  seq_len_prefill, seq_len_ctx_decode, num_decode_tokens):
    prefill_mm   = 2 * active_params * seq_len_prefill
    prefill_attn = 4 * num_layers * hidden_size * (seq_len_prefill ** 2)
    decode_mm    = 2 * active_params * num_decode_tokens
    decode_attn  = 4 * num_layers * hidden_size * seq_len_ctx_decode * num_decode_tokens
    return {
        "prefill_flops":      int(prefill_mm   + prefill_attn),
        "decode_flops":       int(decode_mm    + decode_attn),
        "prefill_attn_flops": int(prefill_attn),
        "decode_attn_flops":  int(decode_attn),
    }


# ---------------------------------------------------------------------------
# KV cache bytes.
# Standard MHA/GQA. MLA models (DeepSeek-V2) need a model-specific override
# since their compressed latent is much smaller — those are handled in the
# bench's per-model post-processing, not here.
# ---------------------------------------------------------------------------
def kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len,
                   batch_size=1, dtype_bytes=2):
    """2 × L × H_kv × d_head × seq_len × batch × dtype. Factor 2 = K and V."""
    return int(2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes)
