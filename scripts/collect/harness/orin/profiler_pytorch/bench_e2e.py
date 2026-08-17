#!/usr/bin/env python3
"""
Profiling Jetson Nano with Pytorch LLM
"""
import json
import sys
import time
import os
import gc
# - max_split_size_mb:128 = limit block splitting to reduce fragmentation
# - expandable_segments:False = don't grow reserved pool
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,expandable_segments:False')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# Compatibility shim for trust_remote_code models (e.g. DeepSeek-V2-Lite) that
# call DynamicCache.get_usable_length(), which transformers ≥ 4.45 renamed to
# get_seq_length. Alias the missing method so the custom modeling code works.
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, seq_length, layer_idx=0: self.get_seq_length(layer_idx)
        )
except ImportError:
    pass

from memory_analysis import analyze_gpu_memory_usage
from power_monitor import TegrastatsMonitor, PowerTracer
from gpu_utils import (
    drop_caches,
    get_mem,
    get_system_memory,
    measure_cuda_context_overhead,
    minimize_memory_pool,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_spec import (  # noqa: E402
    get_spec, compute_flops, peak_tflops_for_quant,
    moe_active_params, kv_cache_bytes,
)


def load_model(model_id: str, quantization: str = "4bit", num_layers: int = None, context_size: int = None, power_tracer=None):
    """Load model with specified quantization and optional layer limit.

    Returns: (model, tokenizer, mem_tokenizer, mem_model)
    """
    print(f"\n{'=' * 50}", file=sys.stderr)
    print(f"Loading Model: {model_id}", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)
    print(f"Quantization: {quantization}", file=sys.stderr)
    if num_layers:
        print(f"Limiting to {num_layers} layer(s)", file=sys.stderr)
    if context_size:
        print(f"Context size: {context_size}", file=sys.stderr)
    else:
        print(f"Context size: using model default", file=sys.stderr)

    # Track memory before tokenizer
    mem_before_tokenizer = get_mem()

    # Sub-phase: Load tokenizer
    if power_tracer:
        power_tracer.mark_phase("load_tokenizer")
    print(f"Loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: {type(tokenizer).__name__}", file=sys.stderr)

    mem_tokenizer = get_mem()

    # Build config kwargs for layer limiting and context size
    config_kwargs = {}
    if num_layers:
        config_kwargs["num_hidden_layers"] = num_layers
    if context_size:
        config_kwargs["max_position_embeddings"] = context_size

    # Clear CUDA cache before loading
    torch.cuda.empty_cache()
    gc.collect()

    # Sub-phase: Load model weights
    if power_tracer:
        power_tracer.mark_phase("load_weights")
    print(f"Loading model...", file=sys.stderr)
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 required for Gemma 3
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,  # Force bf16 for non-quantized layers
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **config_kwargs,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,  # Force bf16 for non-quantized layers
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **config_kwargs,
        )
    elif quantization == "fp32":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **config_kwargs,
        ).to("cuda")
    else:  # bf16 (default)
        # Use explicit .to("cuda") instead of device_map="auto" to avoid accelerate overhead
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **config_kwargs,
        ).to("cuda")

    print(f"Model loaded: {type(model).__name__}", file=sys.stderr)
    print(f"Model device: {model.device}", file=sys.stderr)
    print(f"Model context size (max_position_embeddings): {model.config.max_position_embeddings}", file=sys.stderr)
    print(f"Model num_hidden_layers: {model.config.num_hidden_layers}", file=sys.stderr)

    # Apply torch.compile if requested
    use_compile = os.environ.get('TORCH_COMPILE', '').lower() in ('1', 'true', 'yes')
    if use_compile:
        compile_mode = os.environ.get('TORCH_COMPILE_MODE', 'default')
        compile_dynamic = os.environ.get('TORCH_COMPILE_DYNAMIC', '').lower() in ('1', 'true')
        print(f"Applying torch.compile (mode={compile_mode}, dynamic={compile_dynamic})...", file=sys.stderr)
        model = torch.compile(model, mode=compile_mode, dynamic=compile_dynamic)
        print(f"torch.compile applied", file=sys.stderr)

    # Disable TF32 for better compatibility on Jetson
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model.eval()
    torch.cuda.synchronize()

    # Clean up after model load
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    drop_caches()

    mem_model = get_mem()

    return model, tokenizer, mem_tokenizer, mem_model


def run_e2e_benchmark(
    model_id: str,
    prompt_tokens: int,
    gen_tokens: int,
    quantization: str,
    num_runs: int,
    num_layers: int = None,
    context_size: int = None,
    batch_size: int = 1,
    output_dir: str = None,
) -> dict:
    """
    Run end-to-end benchmark.
    1. Prefill: Process prompt, build KV cache, get first token (TTFT)
    2. Decode: Generate remaining tokens using KV cache (TPOT)
    """
    # =========================================================================
    # POWER TRACER: Start continuous monitoring with phase markers
    # =========================================================================
    power_tracer = PowerTracer(interval_ms=1)
    power_tracer.start()
    power_tracer.mark_phase("idle")

    # =========================================================================
    # STEP 1: Baseline (imports already done)
    # =========================================================================
    mem_baseline = get_mem()

    # =========================================================================
    # STEP 2: CUDA init (should already be initialized from imports)
    # =========================================================================
    _ = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    mem_cuda = get_mem()

    # Measure idle power before loading model
    idle_settle = float(os.environ.get("IDLE_SETTLE_SEC", "0"))
    idle_sample = float(os.environ.get("IDLE_SAMPLE_SEC", "3"))
    if idle_settle > 0:
        print(f"\nIdle settle ({idle_settle:.0f}s)...", file=sys.stderr)
        time.sleep(idle_settle)
    print(f"\nMeasuring idle power ({idle_sample:.0f}s)...", file=sys.stderr)
    idle_monitor = TegrastatsMonitor(interval_ms=5)
    with idle_monitor:
        time.sleep(idle_sample)
    idle_power = idle_monitor.get_power_breakdown()
    print(f"  Idle VDD_IN: {idle_power['vdd_in_mw']} mW, GPU Util: {idle_power['gpu_util_pct']}%, CPU Util: {idle_power['cpu_util_pct']}%", file=sys.stderr)

    # Reset peak memory stats before loading model
    torch.cuda.reset_peak_memory_stats()

    gc.collect()
    drop_caches()

    # =========================================================================
    # STEP 3: Imports (already done at module level)
    # =========================================================================
    mem_imports = get_mem()

    # =========================================================================
    # STEP 4 & 5: Load tokenizer and model
    # =========================================================================
    power_tracer.mark_phase("load")
    model, tokenizer, mem_tokenizer, mem_model = load_model(model_id, quantization, num_layers, context_size, power_tracer)

    # =========================================================================
    # STEP 6: Tokenize prompt (CPU operation)
    # =========================================================================
    base_text = "This is for jetson profiling, let's test it. "
    prompt = base_text * (prompt_tokens // 10)

    tokenize_start = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_tokens)
    tokenize_end = time.perf_counter()
    tokenize_time = tokenize_end - tokenize_start

    mem_tokenize = get_mem()

    # =========================================================================
    # STEP 7: Move tokens to GPU
    # =========================================================================
    to_gpu_start = time.perf_counter()
    input_ids = inputs["input_ids"].to(model.device)
    torch.cuda.synchronize()
    to_gpu_end = time.perf_counter()
    to_gpu_time = to_gpu_end - to_gpu_start

    actual_prompt_tokens = input_ids.shape[1]

    # Replicate for batch size
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)

    torch.cuda.synchronize()
    mem_to_gpu = get_mem()

    print(f"Tokenization: {tokenize_time*1000:.2f}ms, To GPU: {to_gpu_time*1000:.2f}ms", file=sys.stderr)

    print(f"Batch size: {batch_size}, Prompt tokens: {actual_prompt_tokens}, Generate tokens: {gen_tokens}", file=sys.stderr)

    
    # Warmup runs (not timed) - stabilize GPU clocks and JIT compile kernels
    print(f"Warmup runs (3 iterations)...", file=sys.stderr)
    for _ in range(3):
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            del outputs
        torch.cuda.synchronize()
    if os.environ.get('TORCH_COMPILE', '').lower() in ('1', 'true', 'yes'):
        # torch.compile traces/compiles the decode graph (single token +
        # KV cache) on its first decode-shaped call, which the prefill-only
        # warmup above never triggers; without this the ~100 s of tracing
        # lands inside the measured decode window and inflates TPOT ~50x.
        # Eager runs skip this block, so their measurement path is unchanged.
        print(f"Compile warmup: decode path...", file=sys.stderr)
        with torch.no_grad():
            wu_out = model(input_ids, use_cache=True)
            wu_past = wu_out.past_key_values
            wu_tok = torch.argmax(wu_out.logits[:, -1, :], dim=-1, keepdim=True)
            # 10 steps: the SDPA lowering guards on KV-length alignment
            # ((1 + cache_len) % 8), so crossing a full mod-8 cycle here
            # compiles BOTH alignment variants; fewer steps leaves one
            # variant to recompile (~30 s) inside the measured decode.
            for _ in range(10):
                wu_out = model(wu_tok, past_key_values=wu_past, use_cache=True)
                wu_past = wu_out.past_key_values
                wu_tok = torch.argmax(wu_out.logits[:, -1, :], dim=-1, keepdim=True)
            del wu_out, wu_past, wu_tok
        torch.cuda.synchronize()
        print(f"Compile warmup complete.", file=sys.stderr)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)
    print(f"Warmup complete.", file=sys.stderr)

    # print(f"Running {num_runs} end-to-end benchmark runs...", file=sys.stderr)

    # Benchmark runs
    prefill_times = []
    decode_times = []
    total_times = []
    tokens_generated_list = []
    prefill_power_samples = []  # List of power breakdowns
    decode_power_samples = []
    mem_prefill = None  # Track memory after first prefill
    mem_decode = None   # Track memory after first decode

    for i in range(num_runs):
        torch.cuda.synchronize()

        # === PREFILL PHASE ===
        # Process the prompt and get KV cache + first token.
        # perf_counter_ns timestamps bracket the actual GPU work so that
        # get_power_breakdown() only averages tegrastats samples INSIDE the phase.
        prefill_monitor = TegrastatsMonitor(interval_ms=10)
        with prefill_monitor:
            torch.cuda.synchronize()  # Ensure GPU idle before timing
            prefill_start_ns = time.perf_counter_ns()
            prefill_start = time.perf_counter()

            with torch.no_grad():
                # ===== PREFILL_FORWARD =====
                # Run full forward pass through all transformer layers
                # Input: [batch, seq_len] token IDs
                # Output: logits [batch, seq_len, vocab_size] + KV cache
                # Then sample first token: slice logits + argmax
                power_tracer.mark_phase("prefill_forward")
                outputs = model(input_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch, 1]
                torch.cuda.synchronize()

            prefill_end = time.perf_counter()
            prefill_end_ns = time.perf_counter_ns()

        prefill_time = prefill_end - prefill_start
        prefill_power_samples.append(prefill_monitor.get_power_breakdown(prefill_start_ns, prefill_end_ns))

        # Track memory after first prefill (for summary table)
        if i == 0:
            mem_prefill = get_mem()
            # Delete prefill outputs to free logits (we only need next_token and past_key_values)
            del outputs, next_token_logits
            # Clear freed tensors from memory pool before decode
            gc.collect()
            torch.cuda.empty_cache()
            drop_caches()
            mem_prefill_after_clear = get_mem()
            # print(f"  Mem prefill after clear: {mem_prefill_after_clear}", file=sys.stderr)


        # === DECODE PHASE ===
        # Generate remaining tokens using KV cache
        decode_monitor = TegrastatsMonitor(interval_ms=5)
        with decode_monitor:
            torch.cuda.synchronize()  # Ensure GPU idle before timing
            decode_start_ns = time.perf_counter_ns()
            decode_start = time.perf_counter()

            generated_tokens = [next_token]
            current_token = next_token

            with torch.no_grad():
                for token_idx in range(gen_tokens - 1):  # -1 because we already have first token
                    # Mark progress every 10 tokens
                    if token_idx % 10 == 0:
                        power_tracer.mark_phase(f"decode_t{token_idx}")

                    outputs = model(
                        current_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values

                    next_token_logits = outputs.logits[:, -1, :]
                    current_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated_tokens.append(current_token)

                    # Don't stop at EOS — always generate exact requested token count
                    # if batch_size == 1:
                    #     if current_token.item() == tokenizer.eos_token_id:
                    #         break
                    # else:
                    #     if (current_token.squeeze(-1) == tokenizer.eos_token_id).all():
                    #         break

            power_tracer.mark_phase("decode_done")
            torch.cuda.synchronize()
            decode_end = time.perf_counter()
            decode_end_ns = time.perf_counter_ns()

        decode_time = decode_end - decode_start
        decode_power_samples.append(decode_monitor.get_power_breakdown(decode_start_ns, decode_end_ns))

        # Track memory after first decode (for summary table)
        if i == 0:
            mem_decode = get_mem()
            # Clear decode tensors from memory pool
            gc.collect()
            torch.cuda.empty_cache()
            drop_caches()
            mem_decode_after_clear = get_mem()
            # How much memory was freed by the clear step
            sys_dropped = mem_decode['sys_used'] - mem_decode_after_clear.get('sys_used', 0)
            gpu_dropped = mem_decode['gpu_used'] - mem_decode_after_clear.get('gpu_used', 0)

            print(
                f"  Mem decode dropped: "
                f"{sys_dropped:.0f} MB SysRAM, {gpu_dropped:.0f} MB GPU",
                file=sys.stderr,
            )
            # print(f"  Mem decode after clear: {mem_decode_after_clear}", file=sys.stderr)

        total_time = prefill_time + decode_time
        tokens_generated = len(generated_tokens)

        prefill_times.append(prefill_time)
        decode_times.append(decode_time)
        total_times.append(total_time)
        tokens_generated_list.append(tokens_generated)

        # Calculate metrics for this run
        ttft_ms = prefill_time * 1000
        tpot_ms = (decode_time / (tokens_generated - 1) * 1000) if tokens_generated > 1 else 0
        decode_tps = (tokens_generated - 1) / decode_time if decode_time > 0 and tokens_generated > 1 else 0

        print(f"  Run {i+1}: TTFT={ttft_ms:.2f}ms, Decode={tokens_generated-1} tokens @ {decode_tps:.2f} tok/s (TPOT={tpot_ms:.2f}ms)", file=sys.stderr)

        # Clear KV cache and tensors between runs (but keep model loaded)
        del past_key_values, outputs, generated_tokens
        del next_token, current_token, next_token_logits
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate averages
    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decode = sum(decode_times) / len(decode_times)
    avg_total = sum(total_times) / len(total_times)
    avg_tokens_gen = sum(tokens_generated_list) / len(tokens_generated_list)

    # Key metrics (throughput accounts for batch size)
    ttft_ms = avg_prefill * 1000  # Time to first token
    tpot_ms = (avg_decode / (avg_tokens_gen - 1) * 1000) if avg_tokens_gen > 1 else 0  # Time per output token
    prefill_throughput = (actual_prompt_tokens * batch_size) / avg_prefill if avg_prefill > 0 else 0
    decode_throughput = ((avg_tokens_gen - 1) * batch_size) / avg_decode if avg_decode > 0 and avg_tokens_gen > 1 else 0

    # Memory stats
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Capture model config before cleanup for theoretical calculations
    model_config = {
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_hidden_layers': model.config.num_hidden_layers,
        'hidden_size': model.config.hidden_size,
        'num_attention_heads': getattr(model.config, 'num_attention_heads', getattr(model.config, 'num_heads', 8)),
        'num_key_value_heads': getattr(model.config, 'num_key_value_heads', getattr(model.config, 'num_attention_heads', 8)),
        'head_dim': getattr(model.config, 'head_dim', model.config.hidden_size // getattr(model.config, 'num_attention_heads', 8)),
        'max_position_embeddings': model.config.max_position_embeddings,
        'vocab_size': model.config.vocab_size,
        'num_local_experts': getattr(model.config, 'num_local_experts', 0) or getattr(model.config, 'n_routed_experts', 0),
        'num_experts_per_tok': getattr(model.config, 'num_experts_per_tok', 0),
    }

    # FLOPs with attention quadratic term (MLPerf / NVIDIA NeMo convention):
    #   prefill = 2*P*T_pp + 4*L*d_model*T_pp^2
    #   decode  = 2*P*N_dec + 4*L*d_model*T_ctx*N_dec
    spec = get_spec()
    num_params_calc = model_config['num_parameters']
    active_params_calc = moe_active_params(
        num_params=num_params_calc,
        num_layers=model_config['num_hidden_layers'],
        hidden_size=model_config['hidden_size'],
        intermediate_size=getattr(model.config if hasattr(model, 'config') else None, 'intermediate_size',
                                  model_config['hidden_size'] * 4) if False else model_config.get('intermediate_size', model_config['hidden_size'] * 4),
        num_local_experts=model_config['num_local_experts'],
        num_experts_per_tok=model_config['num_experts_per_tok'],
        n_shared_experts=model_config.get('n_shared_experts', 0),
        vocab_size=model_config['vocab_size'],
    )
    avg_decode_tokens_calc = (avg_tokens_gen - 1) if avg_tokens_gen > 1 else 1
    avg_ctx_decode = actual_prompt_tokens + avg_tokens_gen / 2
    flops_calc = compute_flops(
        num_params=num_params_calc, active_params=active_params_calc,
        num_layers=model_config['num_hidden_layers'], hidden_size=model_config['hidden_size'],
        seq_len_prefill=actual_prompt_tokens,
        seq_len_ctx_decode=avg_ctx_decode,
        num_decode_tokens=avg_decode_tokens_calc,
    )
    prefill_tflops = (flops_calc['prefill_flops'] / avg_prefill / 1e12) if avg_prefill > 0 else 0
    decode_tflops = (flops_calc['decode_flops'] / avg_decode / 1e12) if avg_decode > 0 else 0
    # MFU against peak at the actual compute precision used by the kernel.
    # bitsandbytes 4bit/8bit dequant to FP16 before matmul → FP16 peak.
    peak_tflops_used = peak_tflops_for_quant(quantization)
    pp_mfu = (prefill_tflops / peak_tflops_used) if peak_tflops_used else 0
    dec_mfu = (decode_tflops / peak_tflops_used) if peak_tflops_used else 0

    # KV cache bytes for context-aware MBU correction (weights + KV traffic per decode token)
    kv_bytes_prefill = kv_cache_bytes(
        num_layers=model_config['num_hidden_layers'],
        num_kv_heads=model_config['num_key_value_heads'],
        head_dim=model_config['head_dim'],
        seq_len=actual_prompt_tokens, batch_size=batch_size, dtype_bytes=2,
    )
    kv_bytes_avg_decode = kv_cache_bytes(
        num_layers=model_config['num_hidden_layers'],
        num_kv_heads=model_config['num_key_value_heads'],
        head_dim=model_config['head_dim'],
        seq_len=int(avg_ctx_decode), batch_size=batch_size, dtype_bytes=2,
    )

    # =========================================================================
    # STEP 10: Cleanup
    # =========================================================================
    power_tracer.mark_phase("cleanup")
    # Note: outputs, past_key_values, generated_tokens are already deleted in the loop
    # Delete remaining tensors
    del input_ids, inputs

    # Delete model
    del model
    gc.collect()  # Multiple passes
    torch.cuda.empty_cache()

    # Delete tokenizer
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Minimize memory pool (includes drop_caches internally)
    minimize_memory_pool()

    # Sleep to allow async memory release
    time.sleep(1)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    mem_cleanup = get_mem()

    # =========================================================================
    # MEMORY SUMMARY TABLE
    # =========================================================================
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  MEMORY SUMMARY", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    # Calculate theoretical memory values
    num_params = model_config['num_parameters']
    num_layers = model_config['num_hidden_layers']
    num_kv_heads = model_config['num_key_value_heads']
    head_dim = model_config['head_dim']
    hidden_size = model_config['hidden_size']
    vocab_size = model_config['vocab_size']

    # Embedding parameters (NOT quantized, kept in BF16)
    # Input embeddings: vocab_size × hidden_size
    # Output embeddings (lm_head): often shared or same size
    embedding_params = vocab_size * hidden_size
    embedding_mb = (embedding_params * 2) / 1024**2  # BF16 = 2 bytes

    # Non-embedding parameters (quantized)
    non_embedding_params = num_params - embedding_params

    # Bytes per parameter based on quantization (only for non-embedding params)
    if quantization == "4bit":
        bytes_per_param = 0.5
        quant_overhead = 1.1  # quantization state overhead
    elif quantization == "8bit":
        bytes_per_param = 1.0
        quant_overhead = 1.05
    elif quantization == "bf16":
        bytes_per_param = 2.0
        quant_overhead = 1.0
    else:  # fp32
        bytes_per_param = 4.0
        quant_overhead = 1.0

    quantized_mb = (non_embedding_params * bytes_per_param * quant_overhead) / 1024**2
    theory_model_mb = embedding_mb + quantized_mb

    # KV cache per token: 2 (K+V) × num_kv_heads × head_dim × 2 bytes × num_layers
    kv_per_token = 2 * num_kv_heads * head_dim * 2 * num_layers  # bytes
    theory_kv_prefill_mb = (kv_per_token * actual_prompt_tokens * batch_size) / 1024**2

    # Prefill logits tensor: batch × seq × vocab_size × 2 bytes (BF16)
    prefill_logits_mb = (batch_size * actual_prompt_tokens * vocab_size * 2) / 1024**2

    # Decode logits tensor: batch × 1 × vocab_size × 2 bytes (BF16) - much smaller
    decode_logits_mb = (batch_size * 1 * vocab_size * 2) / 1024**2

    # Activation memory for prefill
    # Reference: https://kipp.ly/blog/transformer-inference-arithmetic/
    #            https://blog.eleuther.ai/transformer-math/
    s, b, h, a = actual_prompt_tokens, batch_size, hidden_size, model_config['num_attention_heads']
    L = num_layers

    # Peak activation during inference forward pass (one layer at a time):
    # - Hidden states in/out: 2 × sbh × 2 bytes
    # - MLP intermediate (4× expansion): 4 × sbh × 2 bytes (PEAK - largest tensor)
    # - Attention Q,K,V: 3 × sbh × 2 bytes (computed sequentially)
    # - Attention scores (without FlashAttn): a × s² × b × 2 bytes
    # Peak is typically during MLP: 12 × sbh × 2 bytes
    peak_activation_bytes = 12 * s * b * h * 2

    # Attention score matrix 
    attention_scores_bytes = a * s * s * b * 2

    # BitsAndBytes quantization: weights dequantized to BF16 during forward pass
    # Each layer's weights are dequantized temporarily
    if quantization == "4bit":
        params_per_layer = non_embedding_params / L
        bnb_dequant_bytes = params_per_layer * 2  # BF16 dequantized weights per layer
    elif quantization == "8bit":
        params_per_layer = non_embedding_params / L
        bnb_dequant_bytes = params_per_layer * 2  # BF16 dequantized weights per layer
    else:
        bnb_dequant_bytes = 0

    # Total inference activation
    theory_activation_mb = (peak_activation_bytes + attention_scores_bytes + bnb_dequant_bytes) / 1024**2

    # Total prefill memory: KV cache + logits + activations
    theory_prefill_mb = theory_kv_prefill_mb + prefill_logits_mb + theory_activation_mb

    # KV cache growth per decode token
    theory_kv_decode_mb = (kv_per_token * int(avg_tokens_gen) * batch_size) / 1024**2

    # Decode delta: KV grows, prefill logits freed, decode logits added
    # Net = KV_growth - prefill_logits + decode_logits
    theory_decode_delta_mb = theory_kv_decode_mb - prefill_logits_mb + decode_logits_mb

    stages = [
        ("1. Baseline", mem_baseline, None, "-"),
        ("2. CUDA init", mem_cuda, mem_baseline, "~0"),
        ("3. Imports", mem_imports, mem_cuda, "~0"),
        ("4. Tokenizer", mem_tokenizer, mem_imports, " apprx. 300"),
        ("5. Model", mem_model, mem_tokenizer, f"{theory_model_mb:.0f}"),
        ("6. Tokenize", mem_tokenize, mem_model, "~0"),
        ("7. Tokens to GPU", mem_to_gpu, mem_tokenize, "~0"),
        ("8. Prefill", mem_prefill, mem_to_gpu, f"{385 + theory_prefill_mb:.0f}"),
        ("9. Decode", mem_decode, mem_prefill, f"{theory_decode_delta_mb:.0f}"),
        ("10. Cleanup", mem_cleanup, mem_decode, f"-{theory_model_mb + theory_kv_prefill_mb + theory_kv_decode_mb + decode_logits_mb:.0f}"),
    ]

    print(f"\n  {'Stage':<16} {'GPU':>8} {'PT Alloc':>9} {'PT Rsrvd':>9} {'Non-PT':>8} {'SysRAM':>8} {'Delta':>8} {'Theory':>8}", file=sys.stderr)
    print(f"  {'-'*16} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", file=sys.stderr)

    for name, mem, prev, theory in stages:
        if mem is None:
            continue
        delta = f"+{mem['gpu_used'] - prev['gpu_used']:.0f}" if prev else "-"
        sys_used = mem.get('sys_used', 0)
        print(f"  {name:<16} {mem['gpu_used']:>8.0f} {mem['pytorch_alloc']:>9.0f} {mem['pytorch_reserved']:>9.0f} {mem['non_pytorch']:>8.0f} {sys_used:>8.0f} {delta:>8} {theory:>8}", file=sys.stderr)

    # Print model config and theory breakdown
    print(f"\n  Model Config:", file=sys.stderr)
    print(f"    Total params: {num_params/1e6:.1f}M | Embedding: {embedding_params/1e6:.1f}M | Transformer: {non_embedding_params/1e6:.1f}M", file=sys.stderr)
    print(f"    Layers: {num_layers} | Hidden: {hidden_size} | Attn heads: {a} | KV heads: {num_kv_heads} | Head dim: {head_dim}", file=sys.stderr)
    print(f"    Vocab size: {vocab_size} | Batch: {batch_size} | Seq len: {actual_prompt_tokens}+{int(avg_tokens_gen)}", file=sys.stderr)

    print(f"\n  Theory Breakdown ({quantization}):", file=sys.stderr)
    print(f"    Model:    Embedding={embedding_mb:.0f}MB (BF16) + Quantized={quantized_mb:.0f}MB = {theory_model_mb:.0f}MB", file=sys.stderr)
    print(f"    Prefill:  KV={theory_kv_prefill_mb:.0f}MB + Logits={prefill_logits_mb:.0f}MB + Activations={theory_activation_mb:.0f}MB = {theory_prefill_mb:.0f}MB", file=sys.stderr)
    print(f"    Decode:   KV={theory_kv_decode_mb:.0f}MB - PrefillLogits={prefill_logits_mb:.0f}MB + DecodeLogits={decode_logits_mb:.0f}MB = {theory_decode_delta_mb:.0f}MB", file=sys.stderr)
    print(f"    KV/token: {kv_per_token/1024:.2f}KB × {num_layers} layers", file=sys.stderr)
    print(f"    Activations (inference peak): (10sbh×2 + as²b×2 + bnb_dequant) × 1.2", file=sys.stderr)
    print(f"      = ({peak_activation_bytes/1024**2:.1f}MB + {attention_scores_bytes/1024**2:.1f}MB + {bnb_dequant_bytes/1024**2:.1f}MB)  = {theory_activation_mb:.0f}MB", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    # Average power breakdown across runs
    def avg_breakdown(samples: list, key: str) -> float:
        values = [s.get(key, 0) for s in samples]
        return sum(values) / len(values) if values else 0

    def max_breakdown(samples: list, key: str) -> float:
        """Get max value across all samples for a given key."""
        values = [s.get(key, 0) for s in samples]
        return max(values) if values else 0

    def avg_breakdown_str(samples: list, key: str) -> str:
        """Get the last string value (for RAM use/total which is already formatted)."""
        for s in reversed(samples):
            if key in s:
                return s[key]
        return "0/0MB"

    def _max_bool(samples, key):
        return any(s.get(key, False) for s in samples)

    _pp_dram = int(avg_breakdown(prefill_power_samples, 'dram_mw'))
    _pp_total = int(avg_breakdown(prefill_power_samples, 'total_mw'))
    prefill_power = {
        # Semantic (device-normalized) power fields
        'total_mw':  _pp_total,
        'gpu_mw':    int(avg_breakdown(prefill_power_samples, 'gpu_mw')),
        'cpu_mw':    int(avg_breakdown(prefill_power_samples, 'cpu_mw')),
        'soc_mw':    int(avg_breakdown(prefill_power_samples, 'soc_mw')),
        'dram_mw':   _pp_dram,
        'total4_mw': _pp_total + _pp_dram,
        # Legacy aliases (vdd_in_mw = total; vdd_cpu_gpu_cv_mw = gpu on AGX, shared on Nano)
        'vdd_in_mw': int(avg_breakdown(prefill_power_samples, 'vdd_in_mw')),
        'vdd_cpu_gpu_cv_mw': int(avg_breakdown(prefill_power_samples, 'vdd_cpu_gpu_cv_mw')),
        'vdd_soc_mw': int(avg_breakdown(prefill_power_samples, 'vdd_soc_mw')),
        'gpu_util_pct': int(avg_breakdown(prefill_power_samples, 'gpu_util_pct')),
        'gpu_util_max_pct': int(max_breakdown(prefill_power_samples, 'gpu_util_max_pct')),
        'cpu_util_pct': int(avg_breakdown(prefill_power_samples, 'cpu_util_pct')),
        'cpu_util_max_pct': int(max_breakdown(prefill_power_samples, 'cpu_util_max_pct')),
        'ram_use/total': avg_breakdown_str(prefill_power_samples, 'ram_use/total'),
        'gpu_temp_c': round(avg_breakdown(prefill_power_samples, 'gpu_temp_c'), 1),
        'cpu_temp_c': round(avg_breakdown(prefill_power_samples, 'cpu_temp_c'), 1),
        'samples_avg': round(avg_breakdown(prefill_power_samples, 'samples'), 1),
        'samples_warning': _max_bool(prefill_power_samples, 'samples_warning'),
    }

    _dec_dram = int(avg_breakdown(decode_power_samples, 'dram_mw'))
    _dec_total = int(avg_breakdown(decode_power_samples, 'total_mw'))
    decode_power = {
        'total_mw':  _dec_total,
        'gpu_mw':    int(avg_breakdown(decode_power_samples, 'gpu_mw')),
        'cpu_mw':    int(avg_breakdown(decode_power_samples, 'cpu_mw')),
        'soc_mw':    int(avg_breakdown(decode_power_samples, 'soc_mw')),
        'dram_mw':   _dec_dram,
        'total4_mw': _dec_total + _dec_dram,
        'vdd_in_mw': int(avg_breakdown(decode_power_samples, 'vdd_in_mw')),
        'vdd_cpu_gpu_cv_mw': int(avg_breakdown(decode_power_samples, 'vdd_cpu_gpu_cv_mw')),
        'vdd_soc_mw': int(avg_breakdown(decode_power_samples, 'vdd_soc_mw')),
        'gpu_util_pct': int(avg_breakdown(decode_power_samples, 'gpu_util_pct')),
        'gpu_util_max_pct': int(max_breakdown(decode_power_samples, 'gpu_util_max_pct')),
        'cpu_util_pct': int(avg_breakdown(decode_power_samples, 'cpu_util_pct')),
        'cpu_util_max_pct': int(max_breakdown(decode_power_samples, 'cpu_util_max_pct')),
        'ram_use/total': avg_breakdown_str(decode_power_samples, 'ram_use/total'),
        'gpu_temp_c': round(avg_breakdown(decode_power_samples, 'gpu_temp_c'), 1),
        'cpu_temp_c': round(avg_breakdown(decode_power_samples, 'cpu_temp_c'), 1),
        'samples_avg': round(avg_breakdown(decode_power_samples, 'samples'), 1),
        'samples_warning': _max_bool(decode_power_samples, 'samples_warning'),
    }

    # Calculate energy (power * time) in mJ.
    # Use 4-rail total (3 tegrastats rails + LPDDR5 cell rail VDDQ_VDD2_1V8AO);
    # the DRAM rail adds ~5-10% to total board power and was previously omitted.
    # E_total = P_total_avg * t_phase. Units: mW * s = mJ.
    prefill_energy = prefill_power['total4_mw'] * avg_prefill
    decode_energy  = decode_power['total4_mw']  * avg_decode
    # GPU-attributable energy = (gpu_mw - idle_gpu_mw) * t_phase.
    # On AGX, gpu_mw = VDD_GPU_SOC; on Nano, gpu_mw = VDD_CPU_GPU_CV (shared with CPU).
    idle_gpu_mw = idle_power.get('gpu_mw', 0)
    prefill_gpu_energy = max(prefill_power['gpu_mw'] - idle_gpu_mw, 0) * avg_prefill
    decode_gpu_energy  = max(decode_power['gpu_mw']  - idle_gpu_mw, 0) * avg_decode

    # Memory bandwidth utilization (MBU) — measured (from tegrastats EMC) and roofline-estimate.
    pp_peak_bw = avg_breakdown(prefill_power_samples, 'emc_peak_bw_gb_s') or spec['peak_bw_gb_s']
    dec_peak_bw = avg_breakdown(decode_power_samples, 'emc_peak_bw_gb_s') or spec['peak_bw_gb_s']
    pp_emc_bw = avg_breakdown(prefill_power_samples, 'emc_bw_gb_s')
    dec_emc_bw = avg_breakdown(decode_power_samples, 'emc_bw_gb_s')
    pp_mbu = (pp_emc_bw / pp_peak_bw) if pp_peak_bw else 0
    dec_mbu = (dec_emc_bw / dec_peak_bw) if dec_peak_bw else 0
    # KV-aware roofline estimates (weights once + KV reads per decode token).
    # model_mb is what actually lives on GPU (torch.cuda.max_memory_allocated at prefill).
    # Prefill: reads weights once; KV cache is written (not read from DRAM for Q-K attention).
    # Decode: each token reads weights + KV cache of length T_ctx.
    model_bytes = memory_mb * 1024 * 1024
    est_prefill_bw_gb_s = model_bytes / avg_prefill / 1e9 if avg_prefill > 0 else 0
    dram_traffic_decode = (model_bytes + kv_bytes_avg_decode) * avg_decode_tokens_calc
    est_decode_bw_gb_s = dram_traffic_decode / avg_decode / 1e9 if avg_decode > 0 else 0
    pp_mbu_roofline = (est_prefill_bw_gb_s / pp_peak_bw) if pp_peak_bw else 0
    dec_mbu_roofline = (est_decode_bw_gb_s / dec_peak_bw) if dec_peak_bw else 0

    # =========================================================================
    # POWER TRACER: Stop and save results
    # =========================================================================
    power_tracer.stop()

    # Get phase statistics
    phase_stats = power_tracer.get_phase_stats()
    print(f"\n  Power by Phase:", file=sys.stderr)
    print(f"  {'Phase':<12} {'VDD_IN (mW)':<20} {'VDD_CPU_GPU_CV (mW)':<20} {'Samples'}", file=sys.stderr)
    print(f"  {'-'*12} {'-'*20} {'-'*20} {'-'*8}", file=sys.stderr)
    for phase, stats in sorted(phase_stats.items()):
        vdd_in = f"avg={stats.get('vdd_in_avg_mw', 0)}, max={stats.get('vdd_in_max_mw', 0)}"
        vdd_cgc = f"avg={stats.get('vdd_cpu_gpu_cv_avg_mw', 0)}, max={stats.get('vdd_cpu_gpu_cv_max_mw', 0)}"
        print(f"  {phase:<12} {vdd_in:<20} {vdd_cgc:<20} {stats['samples']}", file=sys.stderr)

    # Save CSV and plot if output_dir is specified
    if output_dir:
        model_name = model_id.replace("/", "_")
        csv_path = os.path.join(output_dir, f"{model_name}_{quantization}_power_trace.csv")
        plot_path = os.path.join(output_dir, f"{model_name}_{quantization}_power_plot.png")

        power_tracer.save_csv(csv_path)
        power_tracer.plot(plot_path, title=f"Power Consumption: {model_id} ({quantization})")

    return {
        "batch_size": batch_size,
        "prompt_tokens": actual_prompt_tokens,
        "generated_tokens": int(avg_tokens_gen),
        "tokenize_time_ms": tokenize_time * 1000,
        "to_gpu_time_ms": to_gpu_time * 1000,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "prefill_throughput_tps": prefill_throughput,
        "decode_throughput_tps": decode_throughput,
        "total_latency_ms": avg_total * 1000,
        "memory_mb": memory_mb,
        # peak_memory_mb is the resident sys-RAM footprint at end of first
        # decode (weights + KV + activation buffers). Other framework benches
        # set this; pytorch was missing the field, so the CSV column had been
        # 0 across all rows. Use mem_decode['sys_used'] to match vLLM's
        # convention. AGX Orin unified memory makes sys_used the meaningful
        # peak number (it covers both GPU and CPU allocations).
        "peak_memory_mb": mem_decode['sys_used'] if mem_decode else (mem_prefill['sys_used'] if mem_prefill else 0),
        "runs": num_runs,
        "prefill_times_ms": [t * 1000 for t in prefill_times],
        "decode_times_ms": [t * 1000 for t in decode_times],
        "idle_power": idle_power,
        "prefill_power": prefill_power,
        "decode_power": decode_power,
        "prefill_energy_mj": prefill_energy,
        "decode_energy_mj": decode_energy,
        "prefill_gpu_energy_mj": prefill_gpu_energy,
        "decode_gpu_energy_mj": decode_gpu_energy,
        "num_params": num_params_calc,
        "active_params": active_params_calc,
        "prefill_tflops": round(prefill_tflops, 4),
        "decode_tflops": round(decode_tflops, 4),
        "prefill_attn_flops": flops_calc['prefill_attn_flops'],
        "decode_attn_flops": flops_calc['decode_attn_flops'],
        "pp_mfu": round(pp_mfu, 4),
        "dec_mfu": round(dec_mfu, 4),
        "pp_mbu_measured": round(pp_mbu, 4),
        "dec_mbu_measured": round(dec_mbu, 4),
        "pp_mbu_roofline": round(pp_mbu_roofline, 4),
        "dec_mbu_roofline": round(dec_mbu_roofline, 4),
        # Legacy keys (kept for existing plotters) — equal *_measured:
        "pp_mbu": round(pp_mbu, 4),
        "dec_mbu": round(dec_mbu, 4),
        "kv_cache_bytes_prefill": kv_bytes_prefill,
        "kv_cache_bytes_decode": kv_bytes_avg_decode,
        "peak_tflops_fp16_dense": spec['peak_tflops_fp16_dense'],
        "peak_tflops_used": peak_tflops_used,
        "peak_bw_gb_s": spec['peak_bw_gb_s'],
        "device_name": spec['name'],
    }


def main():
    if len(sys.argv) < 6:
        print("Usage: bench_e2e.py <model_id> <prompt_tokens> <gen_tokens> <quantization> <runs> [num_layers] [context_size] [batch_size]")
        print("  model_id: HuggingFace model ID (e.g., google/gemma-3-270m-it)")
        print("  prompt_tokens: Number of prompt tokens for prefill")
        print("  gen_tokens: Number of tokens to generate (decode)")
        print("  quantization: 4bit, 8bit, bf16 (default), or fp32")
        print("  runs: Number of benchmark runs")
        print("  num_layers: Optional - limit model to N transformer layers (default: all, use '' to skip)")
        print("  context_size: Optional - max context size (default: model default, use '' to skip)")
        print("  batch_size: Optional - batch size for inference (default: 1)")
        sys.exit(1)

    # Measure CUDA context overhead FIRST (before any other CUDA operations)
    measure_cuda_context_overhead()

    # Drop page caches for consistent benchmark results
    # drop_caches()

    # Analyze GPU memory usage
    analyze_gpu_memory_usage()

    # Parse optional arguments
    num_layers = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
    context_size = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] else None
    batch_size = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] else 1

    # Create output directory for power traces
    output_dir = os.path.join(os.path.dirname(__file__), "results", "power_traces")
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = run_e2e_benchmark(
            model_id=sys.argv[1],
            prompt_tokens=int(sys.argv[2]),
            gen_tokens=int(sys.argv[3]),
            quantization=sys.argv[4],
            num_runs=int(sys.argv[5]),
            num_layers=num_layers,
            context_size=context_size,
            batch_size=batch_size,
            output_dir=output_dir,
        )

        # Output JSON result
        print(json.dumps(result))

        print(f"\n{'=' * 50}", file=sys.stderr)
        print("All tests PASSED!", file=sys.stderr)
        print(f"{'=' * 50}", file=sys.stderr)

    except Exception as e:
        print(f"\nERROR: Benchmark failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Drop page caches after benchmark
    # drop_caches()


if __name__ == "__main__":
    main()
