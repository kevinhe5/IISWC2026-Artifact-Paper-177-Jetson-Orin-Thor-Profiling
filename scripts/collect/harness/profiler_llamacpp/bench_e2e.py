#!/usr/bin/env python3
"""
End-to-end LLM benchmark using llama_cpp: Prefill + Decode with shared KV cache.
Measures real inference performance in a single model load.
Includes power monitoring via tegrastats and detailed memory tracking.
"""
import gc
import json
import os
import sys
import time
from llama_cpp import Llama, llama_synchronize

from power_monitor import TegrastatsMonitor
from gpu_utils import (
    get_mem,
    get_system_memory,
    drop_caches,
    print_memory_status,
    print_memory_summary_table,
    estimate_gguf_memory,
    minimize_memory,
)
from read_gguf import read_metadata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_spec import (  # noqa: E402
    get_spec, compute_flops, peak_tflops_for_quant,
    moe_active_params, kv_cache_bytes,
)


def _quant_name_from_gguf_path(path: str) -> str:
    """Infer quant from GGUF filename suffix; llama.cpp dequants-on-compute → FP16 peak."""
    p = os.path.basename(path).lower()
    for tag in ('q8_0', 'q6_k', 'q5_k', 'q4_k', 'q4_0', 'q3_k', 'q2_k', 'iq4', 'iq3', 'iq2', 'f16', 'bf16'):
        if tag in p:
            return 'gguf_' + tag
    return 'fp16'


def _ensure_int(val, default=1):
    """Ensure value is an integer, handling lists and None."""
    if val is None:
        return default
    if isinstance(val, list):
        return val[0] if val else default
    return int(val)


def calculate_num_params(model_info: dict) -> int:
    """
    Calculate the number of parameters from GGUF model metadata.

    For a typical transformer:
    - Embedding: vocab_size * embed_dim
    - Per layer:
        - Q, K, V projections: embed_dim * (n_heads + 2*n_kv_heads) * head_dim
        - Output projection: n_heads * head_dim * embed_dim
        - FFN: 3 * embed_dim * ffn_dim (gate, up, down)
    - LM head: embed_dim * vocab_size (often tied with embedding)
    """
    # Ensure all values are integers (some GGUF files return lists)
    layers = _ensure_int(model_info.get('layers'), 1)
    embed_dim = _ensure_int(model_info.get('embedding_dim'), 512)
    n_heads = _ensure_int(model_info.get('attention_heads'), 8)
    n_kv_heads = _ensure_int(model_info.get('kv_heads'), n_heads)
    ffn_dim = _ensure_int(model_info.get('ffn_dim'), embed_dim * 4)
    vocab_size = _ensure_int(model_info.get('vocab_size'), 32000)
    moe_experts = _ensure_int(model_info.get('moe_experts'), 0)
    moe_shared = _ensure_int(model_info.get('moe_shared_experts'), 0)
    expert_ffn_dim = _ensure_int(model_info.get('moe_expert_ffn_dim'), ffn_dim) if moe_experts else ffn_dim
    head_dim = embed_dim // n_heads

    # Attention parameters per layer (with GQA)
    q_params = embed_dim * (n_heads * head_dim)       # Q projection
    k_params = embed_dim * (n_kv_heads * head_dim)    # K projection (GQA)
    v_params = embed_dim * (n_kv_heads * head_dim)    # V projection (GQA)
    o_params = (n_heads * head_dim) * embed_dim       # Output projection
    attn_params = q_params + k_params + v_params + o_params

    if moe_experts > 1:
        # MoE: each layer has num_experts × FFN copies (+ shared experts if any).
        ffn_params = (moe_experts + moe_shared) * 3 * embed_dim * expert_ffn_dim
    else:
        # FFN parameters per layer (gate, up, down projections)
        ffn_params = 3 * embed_dim * ffn_dim

    # Total per layer
    params_per_layer = attn_params + ffn_params

    # Embedding + LM head (often tied, count once)
    embedding_params = vocab_size * embed_dim * 2  # input + output embeddings

    total_params = layers * params_per_layer + embedding_params

    return total_params


def run_e2e_benchmark(
    model_path: str,
    ctx_size: int,
    prompt_tokens: int,
    gen_tokens: int,
    gpu_layers: int,
    chunk_size: int,
    num_runs: int
) -> dict:
    """
    Run end-to-end benchmark with proper KV cache sharing.

    This mimics real inference:
    1. Load model once
    2. Prefill: Process prompt tokens, build KV cache (TTFT)
    3. Decode: Generate tokens using KV cache (TPOT)

    Memory is tracked at each stage similar to the PyTorch profiler.
    """
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  llama.cpp End-to-End Benchmark", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    # =========================================================================
    # STEP 1: Baseline memory (before anything)
    # =========================================================================
    gc.collect()
    drop_caches()
    time.sleep(0.5)
    mem_baseline = get_mem()
    print_memory_status("1. Baseline")

    # Estimate model memory from file size
    model_estimate = estimate_gguf_memory(model_path)
    print(f"  Model file: {os.path.basename(model_path)}", file=sys.stderr)
    print(f"  File size: {model_estimate['file_size_mb']:.1f} MB", file=sys.stderr)

    # Read model metadata for TFLOPs calculation
    model_info = read_metadata(model_path)
    num_params = calculate_num_params(model_info)
    print(f"  Parameters: {num_params / 1e6:.1f}M ({num_params / 1e9:.3f}B)", file=sys.stderr)

    # =========================================================================
    # STEP 2: Measure idle power before loading model
    # =========================================================================
    # IDLE_SETTLE_SEC: sleep this long before sampling so transient OS
    #   activity (Docker daemon init, file-cache prefetch, etc.) settles.
    # IDLE_SAMPLE_SEC: sample window length. Longer = lower variance.
    idle_settle = float(os.environ.get("IDLE_SETTLE_SEC", "0"))
    idle_sample = float(os.environ.get("IDLE_SAMPLE_SEC", "3"))
    if idle_settle > 0:
        print(f"\nIdle settle ({idle_settle:.0f}s)...", file=sys.stderr)
        time.sleep(idle_settle)
    print(f"\nMeasuring idle power ({idle_sample:.0f}s)...", file=sys.stderr)
    idle_monitor = TegrastatsMonitor(interval_ms=1)
    with idle_monitor:
        time.sleep(idle_sample)
    idle_power = idle_monitor.get_power_breakdown()
    print(f"  Idle VDD_IN: {idle_power['vdd_in_mw']} mW, GPU Util: {idle_power['gpu_util_pct']}%, CPU Util: {idle_power['cpu_util_pct']}%", file=sys.stderr)

    mem_after_idle = get_mem()
    print_memory_status("2. After idle measurement")

    # =========================================================================
    # STEP 3: Load model
    # =========================================================================
    print(f"\nLoading model: {model_path}", file=sys.stderr)
    print(f"  ctx_size={ctx_size}, gpu_layers={gpu_layers}, chunk_size={chunk_size}", file=sys.stderr)

    # Optional flash-attn opt-in via env var (off by default to match
    # the dustynv container's as-shipped behaviour). The slim-cpp fork's
    # fused RoPE+KV-store op is also gated on flash_attn=True, so any
    # sweep that wants to exercise the fused op must set FLASH_ATTN=1.
    flash_attn = os.environ.get("FLASH_ATTN", "0").strip() in ("1", "true", "yes", "on")
    print(f"  flash_attn={flash_attn}", file=sys.stderr)

    load_start = time.perf_counter()
    model = Llama(
        model_path=model_path,
        n_ctx=ctx_size,
        n_batch=chunk_size,  # n_batch = tokens processed per chunk during prefill
        n_gpu_layers=gpu_layers,
        flash_attn=flash_attn,
        verbose=False
    )
    load_time = time.perf_counter() - load_start
    print(f"  Model loaded in {load_time:.2f}s", file=sys.stderr)

    gc.collect()
    time.sleep(0.5)
    mem_model = get_mem()
    print_memory_status("3. Model loaded")

    # =========================================================================
    # STEP 4: Tokenize prompt (CPU operation)
    # =========================================================================
    tokens = model.tokenize(("Hello " * (prompt_tokens // 2 + 1)).encode())[:prompt_tokens]
    while len(tokens) < prompt_tokens:
        tokens = (tokens * 2)[:prompt_tokens]

    actual_prompt_tokens = len(tokens)
    num_chunks = (actual_prompt_tokens + chunk_size - 1) // chunk_size

    mem_tokenize = get_mem()
    print_memory_status("4. Tokenized")

    print(f"\n  Prompt tokens: {actual_prompt_tokens}, Generate tokens: {gen_tokens}", file=sys.stderr)
    print(f"  Prefill chunks: {num_chunks} (chunk_size={chunk_size})", file=sys.stderr)

    # Warmup runs (not timed) - stabilize GPU clocks and JIT compile kernels
    print(f"  Warmup runs (3 iterations)...", file=sys.stderr)
    for _ in range(3):
        model.reset()
        for chunk_idx in range(0, len(tokens), chunk_size):
            chunk = tokens[chunk_idx:chunk_idx + chunk_size]
            model.eval(chunk)
        # Generate a few tokens
        for _ in range(min(5, gen_tokens)):
            token = model.sample(top_k=1, temp=0.0)
            model.eval([token])
    model.reset()
    gc.collect()
    time.sleep(0.5)
    print(f"  Warmup complete.", file=sys.stderr)

    print(f"  Running {num_runs} end-to-end benchmark runs...", file=sys.stderr)

    # =========================================================================
    # STEP 5-7: Benchmark runs (Prefill + Decode)
    # =========================================================================
    prefill_times = []
    decode_times = []
    total_times = []
    tokens_generated_list = []
    prefill_power_samples = []
    decode_power_samples = []
    mem_prefill = None
    mem_decode = None

    for i in range(num_runs):
        model.reset()
        gc.collect()

        # Memory before prefill (first run only)
        if i == 0:
            mem_before_prefill = get_mem()
            print_memory_status("5. Before prefill")

        # === PREFILL PHASE ===
        # Process prompt tokens in chunks to build KV cache.
        # Phase-window timestamps ensure get_power_breakdown() only averages
        # tegrastats samples captured DURING the phase (not during warmup).
        prefill_monitor = TegrastatsMonitor(interval_ms=1)
        with prefill_monitor:
            prefill_start_ns = time.perf_counter_ns()
            prefill_start = time.perf_counter()

            for chunk_idx in range(0, len(tokens), chunk_size):
                chunk = tokens[chunk_idx:chunk_idx + chunk_size]
                model.eval(chunk)

            # llama.cpp's model.eval() returns after queueing GGML work to the
            # CUDA stream — without an explicit sync, perf_counter captures
            # CPU-side return time, not GPU completion. Same artifact as vLLM's
            # missing torch.cuda.synchronize(): pp_mfu can exceed 1 (physically
            # impossible). This call drains the stream so timing is real.
            try:
                llama_synchronize(model._ctx.ctx)
            except (AttributeError, TypeError):
                pass  # older llama-cpp-python versions

            prefill_end = time.perf_counter()
            prefill_end_ns = time.perf_counter_ns()

        prefill_time = prefill_end - prefill_start
        prefill_power_samples.append(prefill_monitor.get_power_breakdown(prefill_start_ns, prefill_end_ns))

        # Memory after prefill (first run only)
        if i == 0:
            gc.collect()
            time.sleep(0.2)
            mem_prefill = get_mem()
            print_memory_status("6. After prefill")

        # === DECODE PHASE ===
        # Generate tokens one-by-one using the KV cache
        decode_monitor = TegrastatsMonitor(interval_ms=1)
        with decode_monitor:
            decode_start_ns = time.perf_counter_ns()
            decode_start = time.perf_counter()

            generated_count = 0
            for _ in range(gen_tokens):
                token = model.sample(top_k=1, temp=0.0)
                model.eval([token])
                generated_count += 1

                # Don't stop at EOS — always generate exact requested token count
                # if token == model.token_eos():
                #     break

            # Drain the CUDA stream so the final eval() finishes before we stop the clock.
            # (model.sample() inside the loop forces a sync on each iteration since it needs
            # logits, but the very last model.eval() is still in flight when the loop exits.)
            try:
                llama_synchronize(model._ctx.ctx)
            except (AttributeError, TypeError):
                pass

            decode_end = time.perf_counter()
            decode_end_ns = time.perf_counter_ns()

        decode_time = decode_end - decode_start
        decode_power_samples.append(decode_monitor.get_power_breakdown(decode_start_ns, decode_end_ns))

        # Memory after decode (first run only)
        if i == 0:
            gc.collect()
            time.sleep(0.2)
            mem_decode = get_mem()
            print_memory_status("7. After decode")

        total_time = prefill_time + decode_time

        prefill_times.append(prefill_time)
        decode_times.append(decode_time)
        total_times.append(total_time)
        tokens_generated_list.append(generated_count)

        # Calculate metrics for this run
        # TPOT = decode_time / (tokens - 1) since first token is from prefill
        # Reference: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
        ttft_ms = prefill_time * 1000
        decode_tokens = generated_count - 1 if generated_count > 1 else 1
        tpot_ms = (decode_time / decode_tokens * 1000)
        decode_tps = decode_tokens / decode_time if decode_time > 0 else 0

        print(f"    Run {i+1}: TTFT={ttft_ms:.2f}ms, Decode={generated_count} tokens @ {decode_tps:.2f} tok/s (TPOT={tpot_ms:.2f}ms)", file=sys.stderr)

    # Calculate averages
    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decode = sum(decode_times) / len(decode_times)
    avg_total = sum(total_times) / len(total_times)
    avg_tokens_gen = sum(tokens_generated_list) / len(tokens_generated_list)

    # Key metrics
    # TPOT = decode_time / (tokens - 1) since first token is from prefill
    # Reference: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
    ttft_ms = avg_prefill * 1000  # Time to first token
    avg_decode_tokens = avg_tokens_gen - 1 if avg_tokens_gen > 1 else 1
    tpot_ms = (avg_decode / avg_decode_tokens * 1000)  # Time per output token
    prefill_throughput = actual_prompt_tokens / avg_prefill if avg_prefill > 0 else 0
    decode_throughput = avg_decode_tokens / avg_decode if avg_decode > 0 else 0

    # Average power breakdown across runs
    def avg_breakdown(samples: list, key: str) -> float:
        values = [s.get(key, 0) for s in samples]
        return sum(values) / len(values) if values else 0

    def avg_breakdown_str(samples: list, key: str) -> str:
        """Get the last string value (for RAM use/total which is already formatted)."""
        for s in reversed(samples):
            if key in s:
                return s[key]
        return "0/0MB"

    def max_breakdown(samples: list, key: str) -> float:
        """Get max value for a metric across samples."""
        values = [s.get(key, 0) for s in samples]
        return max(values) if values else 0

    def _max_bool(samples, key):
        return any(s.get(key, False) for s in samples)

    def build_power_dict(samples):
        dram_mw = int(avg_breakdown(samples, 'dram_mw'))
        total_mw = int(avg_breakdown(samples, 'total_mw'))
        return {
            'total_mw':  total_mw,
            'gpu_mw':    int(avg_breakdown(samples, 'gpu_mw')),
            'cpu_mw':    int(avg_breakdown(samples, 'cpu_mw')),
            'soc_mw':    int(avg_breakdown(samples, 'soc_mw')),
            'dram_mw':   dram_mw,                      # LPDDR5 cell rail (VDDQ_VDD2_1V8AO), AGX-only
            'total4_mw': total_mw + dram_mw,           # Full 4-rail board power
            'vdd_in_mw':         int(avg_breakdown(samples, 'vdd_in_mw')),
            'vdd_cpu_gpu_cv_mw': int(avg_breakdown(samples, 'vdd_cpu_gpu_cv_mw')),
            'vdd_soc_mw':        int(avg_breakdown(samples, 'vdd_soc_mw')),
            'gpu_util_pct':      int(avg_breakdown(samples, 'gpu_util_pct')),
            'gpu_util_max_pct':  int(max_breakdown(samples, 'gpu_util_max_pct')),
            'cpu_util_pct':      int(avg_breakdown(samples, 'cpu_util_pct')),
            'cpu_util_max_pct':  int(max_breakdown(samples, 'cpu_util_max_pct')),
            'emc_util_pct':      int(avg_breakdown(samples, 'emc_util_pct')),
            'emc_util_max_pct':  int(max_breakdown(samples, 'emc_util_max_pct')),
            'emc_bw_gb_s':       round(avg_breakdown(samples, 'emc_bw_gb_s'), 2),
            'emc_bw_max_gb_s':   round(max_breakdown(samples, 'emc_bw_max_gb_s'), 2),
            'emc_freq_mhz':      int(avg_breakdown(samples, 'emc_freq_mhz')),
            'ram_use/total':     avg_breakdown_str(samples, 'ram_use/total'),
            'gpu_temp_c':        round(avg_breakdown(samples, 'gpu_temp_c'), 1),
            'cpu_temp_c':        round(avg_breakdown(samples, 'cpu_temp_c'), 1),
            'samples_avg':       round(avg_breakdown(samples, 'samples'), 1),
            'samples_warning':   _max_bool(samples, 'samples_warning'),
        }

    prefill_power = build_power_dict(prefill_power_samples)
    decode_power  = build_power_dict(decode_power_samples)

    # Energy = avg total power * phase time. Units: mW * s = mJ.
    # Use 4-rail total (3 tegrastats rails + LPDDR5 DRAM cell rail VDDQ_VDD2_1V8AO).
    # The DRAM rail is hidden from tegrastats but read directly from hwmon sysfs;
    # it adds ~5-10 % to total board power and was previously omitted from energy.
    prefill_energy = prefill_power['total4_mw'] * avg_prefill
    decode_energy  = decode_power['total4_mw']  * avg_decode
    idle_gpu_mw = idle_power.get('gpu_mw', 0)
    prefill_gpu_energy = max(prefill_power['gpu_mw'] - idle_gpu_mw, 0) * avg_prefill
    decode_gpu_energy  = max(decode_power['gpu_mw']  - idle_gpu_mw, 0) * avg_decode

    # Calculate estimated bandwidth (model_size / time)
    model_size_gb = model_estimate['file_size_mb'] / 1024
    # Prefill: load model weights once
    est_prefill_bw = model_size_gb / avg_prefill if avg_prefill > 0 else 0
    # Decode: load model weights per token
    est_decode_bw = (model_size_gb * avg_tokens_gen) / avg_decode if avg_decode > 0 else 0

    # Get EMC frequency and peak bandwidth
    spec = get_spec()
    emc_freq = prefill_power.get('emc_freq_mhz', spec['emc_freq_max_mhz'])
    peak_bw = prefill_power.get('emc_peak_bw_gb_s') or emc_freq * spec['bus_bytes'] * 2 / 1000

    print(f"\n  Bandwidth Analysis:", file=sys.stderr)
    print(f"    Model size: {model_size_gb:.3f} GB", file=sys.stderr)
    print(f"    EMC Frequency: {emc_freq} MHz (Peak BW: {peak_bw:.1f} GB/s)", file=sys.stderr)
    print(f"    Prefill:", file=sys.stderr)
    print(f"      Estimated BW: {est_prefill_bw:.2f} GB/s (model_size / time)", file=sys.stderr)
    print(f"      Actual BW:    {prefill_power['emc_bw_gb_s']:.2f} GB/s ({prefill_power['emc_util_pct']}% util)", file=sys.stderr)
    print(f"    Decode:", file=sys.stderr)
    print(f"      Estimated BW: {est_decode_bw:.2f} GB/s (model_size * {int(avg_tokens_gen)} / time)", file=sys.stderr)
    print(f"      Actual BW:    {decode_power['emc_bw_gb_s']:.2f} GB/s ({decode_power['emc_util_pct']}% util)", file=sys.stderr)

    # FLOPs with attention quadratic term (MLPerf / NVIDIA NeMo convention)
    num_layers      = _ensure_int(model_info.get('layers'), 0)
    hidden_size     = _ensure_int(model_info.get('embedding_dim'), 0)
    num_heads       = _ensure_int(model_info.get('attention_heads'), 8)
    num_kv_heads    = _ensure_int(model_info.get('kv_heads'), num_heads)
    head_dim        = hidden_size // num_heads if num_heads else 0
    intermediate_sz = _ensure_int(model_info.get('ffn_dim'), hidden_size * 4)
    moe_experts     = _ensure_int(model_info.get('moe_experts'), 0)
    moe_top_k       = _ensure_int(model_info.get('moe_top_k'), 0)
    moe_shared      = _ensure_int(model_info.get('moe_shared_experts'), 0)
    expert_ffn_sz   = _ensure_int(model_info.get('moe_expert_ffn_dim'), intermediate_sz) if moe_experts else intermediate_sz
    vocab_size      = _ensure_int(model_info.get('vocab_size'), 32000)
    active_params   = moe_active_params(
        num_params=num_params, num_layers=num_layers, hidden_size=hidden_size,
        intermediate_size=expert_ffn_sz, num_local_experts=moe_experts,
        num_experts_per_tok=moe_top_k, n_shared_experts=moe_shared, vocab_size=vocab_size,
    )
    avg_ctx_decode = actual_prompt_tokens + avg_tokens_gen / 2
    decode_tokens_for_flops = avg_decode_tokens
    flops = compute_flops(
        num_params=num_params, active_params=active_params,
        num_layers=num_layers, hidden_size=hidden_size,
        seq_len_prefill=actual_prompt_tokens,
        seq_len_ctx_decode=avg_ctx_decode,
        num_decode_tokens=decode_tokens_for_flops,
    )
    prefill_flops = flops['prefill_flops']
    decode_flops = flops['decode_flops']
    prefill_tflops = (prefill_flops / avg_prefill / 1e12) if avg_prefill > 0 else 0
    decode_tflops = (decode_flops / avg_decode / 1e12) if avg_decode > 0 else 0
    quant_name = _quant_name_from_gguf_path(model_path)
    peak_tflops_used = peak_tflops_for_quant(quant_name)
    pp_mfu = (prefill_tflops / peak_tflops_used) if peak_tflops_used else 0
    dec_mfu = (decode_tflops / peak_tflops_used) if peak_tflops_used else 0

    # KV-aware decode BW roofline: each decode token reads weights + KV cache.
    # For MoE, only active weights (top_k + shared experts + non-expert) are streamed
    # from DRAM per decode token; the inactive experts stay resident but untouched.
    # active_weight_bytes scales total file size by (active_params / num_params).
    kv_bytes_decode = kv_cache_bytes(
        num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        seq_len=int(avg_ctx_decode), batch_size=1, dtype_bytes=2,
    ) if num_layers else 0
    model_bytes = model_size_gb * 1e9
    active_frac = (active_params / num_params) if num_params > 0 else 1.0
    active_weight_bytes = model_bytes * active_frac
    est_decode_bw_kv = ((active_weight_bytes + kv_bytes_decode) * avg_decode_tokens / avg_decode / 1e9) if avg_decode > 0 else 0
    # Pessimistic "all-weights" roofline kept for comparison (Cohere paper convention B).
    est_decode_bw_total = ((model_bytes + kv_bytes_decode) * avg_decode_tokens / avg_decode / 1e9) if avg_decode > 0 else 0
    pp_mbu_meas = (prefill_power.get('emc_bw_gb_s', 0) / peak_bw) if peak_bw else 0
    dec_mbu_meas = (decode_power.get('emc_bw_gb_s', 0) / peak_bw) if peak_bw else 0
    pp_mbu_roof = (est_prefill_bw / peak_bw) if peak_bw else 0
    dec_mbu_roof = (est_decode_bw_kv / peak_bw) if peak_bw else 0
    dec_mbu_roof_total = (est_decode_bw_total / peak_bw) if peak_bw else 0
    pp_mbu = (est_prefill_bw / peak_bw) if peak_bw else 0
    dec_mbu = (est_decode_bw / peak_bw) if peak_bw else 0

    print(f"\n  Compute Throughput:", file=sys.stderr)
    print(f"    Parameters: {num_params / 1e6:.1f}M", file=sys.stderr)
    print(f"    Prefill TFLOPs: {prefill_tflops:.3f} (2 * {num_params/1e6:.0f}M * {actual_prompt_tokens} / {avg_prefill:.3f}s)", file=sys.stderr)
    print(f"    Decode TFLOPs:  {decode_tflops:.3f} (2 * {num_params/1e6:.0f}M * {int(decode_tokens_for_flops)} / {avg_decode:.3f}s)", file=sys.stderr)

    # =========================================================================
    # STEP 8: Cleanup - delete model and measure memory release
    # =========================================================================
    del model
    del tokens
    gc.collect()
    drop_caches()
    time.sleep(1)  # Allow async memory release

    mem_cleanup = get_mem()
    print_memory_status("8. Cleanup")

    # =========================================================================
    # MEMORY SUMMARY TABLE
    # =========================================================================
    stages = [
        ("1. Baseline", mem_baseline, None, "-"),
        ("2. After idle", mem_after_idle, mem_baseline, "~0"),
        ("3. Model loaded", mem_model, mem_after_idle, f"~{model_estimate['file_size_mb']:.0f}"),
        ("4. Tokenized", mem_tokenize, mem_model, "~0"),
        ("5. Before prefill", mem_before_prefill, mem_tokenize, "~0"),
        ("6. After prefill", mem_prefill, mem_before_prefill, f"+KV({actual_prompt_tokens})"),
        ("7. After decode", mem_decode, mem_prefill, f"+KV({int(avg_tokens_gen)})"),
        ("8. Cleanup", mem_cleanup, mem_decode, f"-model"),
    ]

    print_memory_summary_table(stages)

    # Calculate memory metrics
    model_memory_mb = mem_model['sys_used'] - mem_baseline['sys_used']
    peak_memory_mb = mem_decode['sys_used']
    prefill_memory_delta = mem_prefill['sys_used'] - mem_before_prefill['sys_used'] if mem_prefill and mem_before_prefill else 0
    decode_memory_delta = mem_decode['sys_used'] - mem_prefill['sys_used'] if mem_decode and mem_prefill else 0
    cleanup_freed = mem_decode['sys_used'] - mem_cleanup['sys_used'] if mem_decode else 0

    print(f"\n  Memory Analysis:", file=sys.stderr)
    print(f"    Model load: +{model_memory_mb:.0f} MB (file: {model_estimate['file_size_mb']:.0f} MB)", file=sys.stderr)
    print(f"    Prefill KV: +{prefill_memory_delta:.0f} MB ({actual_prompt_tokens} tokens)", file=sys.stderr)
    print(f"    Decode KV:  +{decode_memory_delta:.0f} MB ({int(avg_tokens_gen)} tokens)", file=sys.stderr)
    print(f"    Cleanup:    -{cleanup_freed:.0f} MB freed", file=sys.stderr)
    print(f"    Peak usage: {peak_memory_mb:.0f} MB", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    return {
        "prompt_tokens": actual_prompt_tokens,
        "generated_tokens": int(avg_tokens_gen),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "prefill_throughput_tps": prefill_throughput,
        "decode_throughput_tps": decode_throughput,
        "total_latency_ms": avg_total * 1000,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "runs": num_runs,
        "load_time_s": load_time,
        "prefill_times_ms": [t * 1000 for t in prefill_times],
        "decode_times_ms": [t * 1000 for t in decode_times],
        "idle_power": idle_power,
        "prefill_power": prefill_power,
        "decode_power": decode_power,
        "prefill_energy_mj": prefill_energy,
        "decode_energy_mj": decode_energy,
        "prefill_gpu_energy_mj": prefill_gpu_energy,
        "decode_gpu_energy_mj": decode_gpu_energy,
        "model_size_gb": model_size_gb,
        "emc_freq_mhz": emc_freq,
        "emc_peak_bw_gb_s": peak_bw,
        "prefill_est_bw_gb_s": round(est_prefill_bw, 2),
        "decode_est_bw_gb_s": round(est_decode_bw, 2),
        "num_params": num_params,
        "active_params": active_params,
        "prefill_tflops": round(prefill_tflops, 4),
        "decode_tflops": round(decode_tflops, 4),
        "prefill_attn_flops": flops['prefill_attn_flops'],
        "decode_attn_flops": flops['decode_attn_flops'],
        "pp_mfu": round(pp_mfu, 4),
        "dec_mfu": round(dec_mfu, 4),
        "pp_mbu_measured": round(pp_mbu_meas, 4),
        "dec_mbu_measured": round(dec_mbu_meas, 4),
        "pp_mbu_roofline": round(pp_mbu_roof, 4),
        "dec_mbu_roofline": round(dec_mbu_roof, 4),
        "dec_mbu_roofline_total": round(dec_mbu_roof_total, 4),
        "active_weight_bytes": int(active_weight_bytes),
        # legacy:
        "pp_mbu": round(pp_mbu, 4),
        "dec_mbu": round(dec_mbu, 4),
        "kv_cache_bytes_decode": kv_bytes_decode,
        "peak_tflops_used": peak_tflops_used,
        "peak_tflops_fp16_dense": spec['peak_tflops_fp16_dense'],
        "peak_bw_gb_s": spec['peak_bw_gb_s'],
        "device_name": spec['name'],
        "quantization_detected": quant_name,
        "memory_mb": model_memory_mb,
        "peak_memory_mb": peak_memory_mb,
        "memory_trace": {
            "baseline_mb": mem_baseline['sys_used'],
            "after_model_mb": mem_model['sys_used'],
            "after_prefill_mb": mem_prefill['sys_used'] if mem_prefill else 0,
            "after_decode_mb": mem_decode['sys_used'] if mem_decode else 0,
            "after_cleanup_mb": mem_cleanup['sys_used'],
            "total_ram_mb": mem_baseline['sys_total'],
            "model_delta_mb": model_memory_mb,
            "prefill_delta_mb": prefill_memory_delta,
            "decode_delta_mb": decode_memory_delta,
            "cleanup_freed_mb": cleanup_freed,
            "gpu_used_mb": mem_decode['gpu_used'] if mem_decode else 0,
            "gpu_total_mb": mem_decode['gpu_total'] if mem_decode else 0,
        },
    }


def main():
    if len(sys.argv) < 8:
        print("Usage: bench_e2e.py <model> <ctx> <prompt_tokens> <gen_tokens> <gpu_layers> <chunk> <runs>")
        print("  model: Path to GGUF model file")
        print("  ctx: Context size")
        print("  prompt_tokens: Number of prompt tokens for prefill")
        print("  gen_tokens: Number of tokens to generate (decode)")
        print("  gpu_layers: Number of layers to offload to GPU")
        print("  chunk: Chunk size (tokens per forward pass during prefill)")
        print("  runs: Number of benchmark runs")
        sys.exit(1)

    result = run_e2e_benchmark(
        model_path=sys.argv[1],
        ctx_size=int(sys.argv[2]),
        prompt_tokens=int(sys.argv[3]),
        gen_tokens=int(sys.argv[4]),
        gpu_layers=int(sys.argv[5]),
        chunk_size=int(sys.argv[6]),
        num_runs=int(sys.argv[7])
    )

    # Output JSON result
    print(json.dumps(result))


if __name__ == "__main__":
    main()
