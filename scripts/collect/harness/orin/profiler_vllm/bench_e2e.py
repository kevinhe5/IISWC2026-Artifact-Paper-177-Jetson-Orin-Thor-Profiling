#!/usr/bin/env python3
"""
End-to-end LLM benchmark using vLLM: Prefill + Decode with power monitoring.
Outputs same CSV schema as llama.cpp profiler for cross-framework comparison.
"""
import gc
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

from power_monitor import TegrastatsMonitor
from gpu_utils import get_mem, get_system_memory, drop_caches, print_memory_status, print_memory_summary_table, minimize_memory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_spec import (  # noqa: E402
    get_spec, compute_flops, peak_tflops_for_quant,
    moe_active_params, kv_cache_bytes,
)


def get_arch_from_config(model_path: str) -> dict:
    """Pull full arch from HF config.json (num_layers, hidden_size, kv, MoE, quant)."""
    EMPTY = {
        'num_layers': 0, 'hidden_size': 0, 'num_kv_heads': 0, 'head_dim': 0,
        'intermediate_size': 0, 'vocab_size': 0,
        'moe_experts': 0, 'moe_top_k': 0, 'n_shared_experts': 0,
        'torch_dtype': 'float16',
    }
    if os.path.isfile(model_path):
        return EMPTY  # GGUF: vLLM extracts from the file directly
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return EMPTY
    with open(config_path) as f:
        cfg = json.load(f)
    hidden = cfg.get('hidden_size', 0)
    n_heads = cfg.get('num_attention_heads', 0)
    return {
        'num_layers':        cfg.get('num_hidden_layers', 0),
        'hidden_size':       hidden,
        'num_kv_heads':      cfg.get('num_key_value_heads', n_heads),
        'head_dim':          cfg.get('head_dim', (hidden // n_heads) if n_heads else 0),
        'intermediate_size': cfg.get('intermediate_size', hidden * 4),
        'vocab_size':        cfg.get('vocab_size', 32000),
        'moe_experts':       cfg.get('num_local_experts', 0) or cfg.get('n_routed_experts', 0),
        'moe_top_k':         cfg.get('num_experts_per_tok', 0),
        'n_shared_experts':  cfg.get('n_shared_experts', 0) or 0,
        'torch_dtype':       (cfg.get('torch_dtype') or 'float16').lower(),
    }


def _quant_name_from_path(model_path: str, arch: dict) -> str:
    """Infer compute precision from HF dtype or GGUF filename suffix."""
    p = os.path.basename(model_path).lower()
    for tag in ('q8_0', 'q6_k', 'q5_k', 'q4_k', 'q4_0', 'q3_k', 'iq4', 'iq3', 'iq2'):
        if tag in p:
            return 'gguf_' + tag   # → FP16 peak (dequant on compute)
    dt = arch.get('torch_dtype', 'float16')
    if dt in ('bfloat16', 'bf16'): return 'bf16'
    if dt in ('float16', 'fp16', 'half'): return 'fp16'
    return 'fp16'


def get_model_num_params(model_path: str) -> int:
    """Get parameter count from HuggingFace model config."""
    if os.path.isfile(model_path):
        # GGUF file — estimate from file size (rough: ~2 bytes per param for FP16)
        size_bytes = os.path.getsize(model_path)
        return size_bytes // 2  # rough estimate
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return 0
    with open(config_path) as f:
        cfg = json.load(f)

    hidden = cfg.get("hidden_size", 0)
    layers = cfg.get("num_hidden_layers", 0)
    n_heads = cfg.get("num_attention_heads", 0)
    n_kv_heads = cfg.get("num_key_value_heads", n_heads)
    intermediate = cfg.get("intermediate_size", hidden * 4)
    vocab = cfg.get("vocab_size", 32000)
    head_dim = hidden // n_heads if n_heads else 0

    attn_per_layer = hidden * (n_heads + 2 * n_kv_heads) * head_dim + n_heads * head_dim * hidden
    ffn_per_layer = 3 * hidden * intermediate
    embedding = vocab * hidden * 2
    return layers * (attn_per_layer + ffn_per_layer) + embedding


def get_model_size_mb(model_path: str) -> float:
    """Estimate model size on disk in MB. Follows symlinks (HF cache)."""
    # Single file (e.g., GGUF)
    if os.path.isfile(model_path):
        return os.path.getsize(os.path.realpath(model_path)) / 1024 / 1024
    total = 0
    for f in os.listdir(model_path):
        if f.endswith(('.safetensors', '.bin', '.pt', '.gguf')):
            fp = os.path.join(model_path, f)
            try:
                total += os.path.getsize(os.path.realpath(fp))
            except OSError:
                pass
    return total / 1024 / 1024


def run_e2e_benchmark(model_path: str, prompt_tokens: int, gen_tokens: int, num_runs: int) -> dict:
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  vLLM End-to-End Benchmark", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    num_params = get_model_num_params(model_path)
    model_size_mb = get_model_size_mb(model_path)
    model_size_gb = model_size_mb / 1024
    print(f"  Model: {os.path.basename(model_path)}", file=sys.stderr)
    print(f"  Size: {model_size_mb:.0f} MB, Params: {num_params/1e6:.1f}M", file=sys.stderr)

    # Baseline memory
    gc.collect()
    drop_caches()
    time.sleep(0.5)
    mem_baseline = get_mem()
    print_memory_status("1. Baseline")

    # Idle power
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

    # Load model.
    # max_model_len must accommodate the max (prompt + gen) in our sweep grid;
    # at 2048 vLLM was silently capping decode for pp+gen>2048 (Llama-1B early-stopped
    # at ~1980 tokens). Sweep grid maxes at pp=4096+gen=4096=8192, so size to fit.
    # gpu_memory_utilization: 0.5 (16 GB) is enough for Llama-1B/8B but Mixtral-Q3
    # is 21 GB on disk — silently OOMs at 0.5. Bump to 0.85 (27 GB) when the model
    # file exceeds ~12 GB so MoE fits while small models still leave room for OS.
    print(f"\nLoading vLLM model...", file=sys.stderr)
    load_start = time.perf_counter()
    max_len = max(prompt_tokens + gen_tokens + 128, 8192)
    # Compute model size — handles both GGUF single file and HF safetensors directory
    try:
        if os.path.isfile(model_path):
            model_size_gb = os.path.getsize(model_path) / 1e9
        elif os.path.isdir(model_path):
            total = 0
            for root, _, files in os.walk(model_path):
                for f in files:
                    if f.endswith(('.safetensors', '.bin', '.gguf', '.pt')):
                        try:
                            total += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
            model_size_gb = total / 1e9
        else:
            model_size_gb = 0
    except OSError:
        model_size_gb = 0
    gpu_mem_util = 0.85 if model_size_gb >= 12 else 0.5
    # Long-context KV pre-allocation: at max_len > 32K the KV pool needs
    # multiple GB pre-allocated up front. The 0.5 default leaves only ~16 GB
    # which is technically enough at 1B/Q4 but vLLM's allocator becomes
    # conservative and silently fails. Bump when context is long.
    if max_len > 32768:
        gpu_mem_util = 0.85
    # Explicit env-var override always wins.
    env_util = os.environ.get("VLLM_GPU_MEM_UTIL", "")
    if env_util:
        try: gpu_mem_util = float(env_util)
        except ValueError: pass
    # If the requested context exceeds the model's max_position_embeddings,
    # vLLM rejects with ValueError unless VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is
    # set. Llama-3.2-1B max is 131072; our gen=131072 + pp=128 + slack overflows
    # by ~256 tokens. Auto-set the env var so the bench at the model's edge
    # context still runs (RoPE supports the position; vLLM just gates it).
    if max_len > 100000:
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    # Optional cache-disable for the no-cache scaling study. Default behavior
    # is unchanged (V1 fp16 caches by default; V0 GGUF doesn't). Setting
    # VLLM_DISABLE_PREFIX_CACHE=1 forces enable_prefix_caching=False on V1.
    disable_cache = os.environ.get("VLLM_DISABLE_PREFIX_CACHE", "0") == "1"
    print(f"  model_size={model_size_gb:.1f} GB → gpu_memory_utilization={gpu_mem_util}"
          f"{' (prefix_caching=OFF)' if disable_cache else ''}", file=sys.stderr)
    # VLLM_ENFORCE_EAGER=0 → CUDA-graph capture on (both V0 and V1).
    # Default remains eager=True to match the master sweep.
    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "1") == "1"
    llm_kwargs = dict(model=model_path, dtype="float16", max_model_len=max_len,
                      enforce_eager=enforce_eager, gpu_memory_utilization=gpu_mem_util,
                      trust_remote_code=True)
    # For batch-1 (edge) workloads, V0's default cudagraph capture list has
    # 39 entries (1..256) which blows Orin's memory budget. Cap it via
    # max_num_seqs. VLLM_MAX_NUM_SEQS env var (default 1) restricts the
    # capture set to at most that many concurrent sequences.
    env_seqs = os.environ.get("VLLM_MAX_NUM_SEQS", "")
    if env_seqs:
        try:
            llm_kwargs["max_num_seqs"] = int(env_seqs)
            # Also shrink CPU swap for V0 to avoid wasted pinned memory
            llm_kwargs["swap_space"] = float(os.environ.get("VLLM_SWAP_SPACE_GB", "1"))
        except ValueError: pass
    if disable_cache:
        llm_kwargs["enable_prefix_caching"] = False
    llm = LLM(**llm_kwargs)
    load_time = time.perf_counter() - load_start
    print(f"  Loaded in {load_time:.2f}s", file=sys.stderr)

    gc.collect()
    time.sleep(0.5)
    mem_model = get_mem()
    print_memory_status("2. Model loaded")

    # Build prompt. Default uses a deterministic "Hello " repetition so that
    # repeated calls hit the prefix cache (matches real-world chat deployment).
    # Setting UNIQUE_PROMPT=1 prepends a random token sequence to force unique
    # block hashes — used by the no-cache scaling study to isolate true
    # prefill latency from cache-hit timing.
    unique_prompt = os.environ.get("UNIQUE_PROMPT", "0") == "1"
    if unique_prompt:
        import secrets
        # Random hex string → tokenizes to varied token IDs. Long enough that
        # the first 16-token block hash differs across cells.
        prompt = secrets.token_hex(64) + " " + ("Hello " * (prompt_tokens // 2 + 1))
    else:
        prompt = "Hello " * (prompt_tokens // 2 + 1)
    # Verify token count
    tokenizer = llm.get_tokenizer()
    tokens = tokenizer.encode(prompt)[:prompt_tokens]
    prompt = tokenizer.decode(tokens)
    actual_prompt_tokens = len(tokens)

    print(f"  Prompt tokens: {actual_prompt_tokens}, Gen tokens: {gen_tokens}", file=sys.stderr)

    # Warmup
    print("  Warmup (3 runs)...", file=sys.stderr)
    warmup_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=5)
    for _ in range(3):
        # use_tqdm=False suppresses vLLM's per-call progress bar (~80 ms render
        # cost on Jetson stderr). Without this, the timer below picks up the
        # render time and reports inflated TTFT — confirmed 2026-04-29 by
        # bench-vs-isolated-test diff.
        llm.generate([prompt], warmup_params, use_tqdm=False)
    print("  Warmup done.", file=sys.stderr)

    # Benchmark runs — two-call approach for accurate TTFT measurement
    # Call 1: generate 1 token → measures prefill (TTFT)
    # Call 2: generate all tokens → total time; decode = total - TTFT
    prefill_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=1, ignore_eos=True)
    decode_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=gen_tokens, ignore_eos=True)

    prefill_times = []
    decode_times = []
    total_times = []
    tokens_generated_list = []
    prefill_power_samples = []
    decode_power_samples = []
    mem_prefill = None
    mem_decode = None

    print(f"  Running {num_runs} benchmark runs (two-call approach)...", file=sys.stderr)
    for i in range(num_runs):
        gc.collect()

        # === PREFILL PHASE ===
        # Generate exactly 1 token to measure TTFT (prefill + first token).
        # Approach B: explicit torch.cuda.synchronize() after generate() — vLLM's
        # high-level API returns when the scheduler considers the request fulfilled,
        # not when the GPU stream has drained. Without sync, TTFT is artificially short.
        import torch as _torch_b  # local import to avoid hard dep at import time
        prefill_monitor = TegrastatsMonitor(interval_ms=1)
        # Drain any pending GPU work from prior warmup calls before timing —
        # without this, vLLM's V1 scheduler might still have queued ops on
        # the stream and the next generate()'s wall absorbs that drain.
        if _torch_b.cuda.is_available():
            _torch_b.cuda.synchronize()
        with prefill_monitor:
            prefill_start_ns = time.perf_counter_ns()
            prefill_start = time.perf_counter()
            llm.generate([prompt], prefill_params, use_tqdm=False)
            if _torch_b.cuda.is_available():
                _torch_b.cuda.synchronize()
            prefill_end = time.perf_counter()
            prefill_end_ns = time.perf_counter_ns()

        prefill_time = prefill_end - prefill_start

        # === DECODE PHASE ===
        # Generate full sequence; decode_time = total - prefill
        decode_monitor = TegrastatsMonitor(interval_ms=1)
        with decode_monitor:
            decode_start_ns = time.perf_counter_ns()
            full_start = time.perf_counter()
            outputs = llm.generate([prompt], decode_params, use_tqdm=False)
            if _torch_b.cuda.is_available():
                _torch_b.cuda.synchronize()
            full_end = time.perf_counter()
            decode_end_ns = time.perf_counter_ns()

        full_time = full_end - full_start
        output = outputs[0]
        generated_count = len(output.outputs[0].token_ids)
        decode_time = full_time - prefill_time
        if decode_time < 0:
            decode_time = full_time * 0.95  # Fallback: assume prefill is ~5%

        total_time = full_time

        prefill_times.append(prefill_time)
        decode_times.append(decode_time)
        total_times.append(total_time)
        tokens_generated_list.append(generated_count)

        # Separate power data per phase, filtered to the actual phase window
        prefill_power_samples.append(prefill_monitor.get_power_breakdown(prefill_start_ns, prefill_end_ns))
        decode_power_samples.append(decode_monitor.get_power_breakdown(decode_start_ns, decode_end_ns))

        if i == 0:
            gc.collect()
            time.sleep(0.2)
            mem_prefill = get_mem()
            mem_decode = get_mem()

        decode_tokens = generated_count - 1 if generated_count > 1 else 1
        decode_tps = decode_tokens / decode_time if decode_time > 0 else 0
        print(f"    Run {i+1}: TTFT={prefill_time*1000:.2f}ms, Decode={generated_count} tokens @ {decode_tps:.1f} tok/s", file=sys.stderr)

    # Averages
    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decode = sum(decode_times) / len(decode_times)
    avg_total = sum(total_times) / len(total_times)
    avg_tokens_gen = sum(tokens_generated_list) / len(tokens_generated_list)

    ttft_ms = avg_prefill * 1000
    avg_decode_tokens = avg_tokens_gen - 1 if avg_tokens_gen > 1 else 1
    tpot_ms = (avg_decode / avg_decode_tokens * 1000)
    prefill_throughput = actual_prompt_tokens / avg_prefill if avg_prefill > 0 else 0
    decode_throughput = avg_decode_tokens / avg_decode if avg_decode > 0 else 0

    # Power averages
    def avg_bd(samples, key):
        vals = [s.get(key, 0) for s in samples]
        return sum(vals) / len(vals) if vals else 0

    def max_bd(samples, key):
        vals = [s.get(key, 0) for s in samples]
        return max(vals) if vals else 0

    def avg_bd_str(samples, key):
        for s in reversed(samples):
            if key in s:
                return s[key]
        return "0/0MB"

    def _max_bool(samples, key):
        return any(s.get(key, False) for s in samples)

    def build_power_dict(samples):
        dram_mw = int(avg_bd(samples, 'dram_mw'))
        total_mw = int(avg_bd(samples, 'total_mw'))
        return {
            'total_mw':  total_mw,
            'gpu_mw':    int(avg_bd(samples, 'gpu_mw')),
            'cpu_mw':    int(avg_bd(samples, 'cpu_mw')),
            'soc_mw':    int(avg_bd(samples, 'soc_mw')),
            'dram_mw':   dram_mw,
            'total4_mw': total_mw + dram_mw,
            'vdd_in_mw':         int(avg_bd(samples, 'vdd_in_mw')),
            'vdd_cpu_gpu_cv_mw': int(avg_bd(samples, 'vdd_cpu_gpu_cv_mw')),
            'vdd_soc_mw':        int(avg_bd(samples, 'vdd_soc_mw')),
            'gpu_util_pct':      int(avg_bd(samples, 'gpu_util_pct')),
            'gpu_util_max_pct':  int(max_bd(samples, 'gpu_util_max_pct')),
            'cpu_util_pct':      int(avg_bd(samples, 'cpu_util_pct')),
            'cpu_util_max_pct':  int(max_bd(samples, 'cpu_util_max_pct')),
            'emc_util_pct':      int(avg_bd(samples, 'emc_util_pct')),
            'emc_util_max_pct':  int(max_bd(samples, 'emc_util_max_pct')),
            'emc_bw_gb_s':       round(avg_bd(samples, 'emc_bw_gb_s'), 2),
            'emc_bw_max_gb_s':   round(max_bd(samples, 'emc_bw_max_gb_s'), 2),
            'emc_freq_mhz':      int(avg_bd(samples, 'emc_freq_mhz')),
            'ram_use/total':     avg_bd_str(samples, 'ram_use/total'),
            'gpu_temp_c':        round(avg_bd(samples, 'gpu_temp_c'), 1),
            'cpu_temp_c':        round(avg_bd(samples, 'cpu_temp_c'), 1),
            'samples_avg':       round(avg_bd(samples, 'samples'), 1),
            'samples_warning':   _max_bool(samples, 'samples_warning'),
        }

    prefill_power = build_power_dict(prefill_power_samples)
    decode_power  = build_power_dict(decode_power_samples)

    # Energy = avg total power * phase time. Units: mW * s = mJ.
    # 4-rail total = 3 tegrastats rails + LPDDR5 cell rail (VDDQ_VDD2_1V8AO).
    prefill_energy = prefill_power['total4_mw'] * avg_prefill
    decode_energy  = decode_power['total4_mw']  * avg_decode
    idle_gpu_mw = idle_power.get('gpu_mw', 0)
    prefill_gpu_energy = max(prefill_power['gpu_mw'] - idle_gpu_mw, 0) * avg_prefill
    decode_gpu_energy  = max(decode_power['gpu_mw']  - idle_gpu_mw, 0) * avg_decode

    spec = get_spec()
    emc_freq = prefill_power.get('emc_freq_mhz', spec['emc_freq_max_mhz'])
    peak_bw = prefill_power.get('emc_peak_bw_gb_s') or emc_freq * spec['bus_bytes'] * 2 / 1000

    arch = get_arch_from_config(model_path)
    active_params = moe_active_params(
        num_params=num_params, num_layers=arch['num_layers'], hidden_size=arch['hidden_size'],
        intermediate_size=arch['intermediate_size'], num_local_experts=arch['moe_experts'],
        num_experts_per_tok=arch['moe_top_k'], n_shared_experts=arch['n_shared_experts'],
        vocab_size=arch['vocab_size'],
    )
    avg_ctx_decode = actual_prompt_tokens + avg_tokens_gen / 2
    flops = compute_flops(
        num_params=num_params, active_params=active_params,
        num_layers=arch['num_layers'], hidden_size=arch['hidden_size'],
        seq_len_prefill=actual_prompt_tokens,
        seq_len_ctx_decode=avg_ctx_decode,
        num_decode_tokens=avg_decode_tokens,
    )
    prefill_flops = flops['prefill_flops']
    decode_flops = flops['decode_flops']
    prefill_tflops = (prefill_flops / avg_prefill / 1e12) if avg_prefill > 0 else 0
    decode_tflops = (decode_flops / avg_decode / 1e12) if avg_decode > 0 else 0
    quant_name = _quant_name_from_path(model_path, arch)
    peak_tflops_used = peak_tflops_for_quant(quant_name)
    pp_mfu = (prefill_tflops / peak_tflops_used) if peak_tflops_used else 0
    dec_mfu = (decode_tflops / peak_tflops_used) if peak_tflops_used else 0

    kv_bytes_decode = kv_cache_bytes(
        num_layers=arch['num_layers'], num_kv_heads=arch['num_kv_heads'],
        head_dim=arch['head_dim'], seq_len=int(avg_ctx_decode), batch_size=1, dtype_bytes=2,
    ) if arch['num_layers'] else 0
    model_bytes = model_size_gb * 1e9
    est_prefill_bw = (model_bytes / avg_prefill / 1e9) if avg_prefill > 0 else 0
    est_decode_bw  = ((model_bytes + kv_bytes_decode) * avg_decode_tokens / avg_decode / 1e9) if avg_decode > 0 else 0
    pp_mbu = (est_prefill_bw / peak_bw) if peak_bw else 0
    dec_mbu = (est_decode_bw / peak_bw) if peak_bw else 0

    # Cleanup
    del llm
    gc.collect()
    drop_caches()
    time.sleep(1)
    mem_cleanup = get_mem()

    model_memory_mb = mem_model['sys_used'] - mem_baseline['sys_used']
    peak_memory_mb = mem_decode['sys_used'] if mem_decode else mem_model['sys_used']

    print(f"\n  Summary: TTFT={ttft_ms:.1f}ms, TPOT={tpot_ms:.2f}ms, Decode={decode_throughput:.1f} tok/s", file=sys.stderr)

    return {
        "prompt_tokens": actual_prompt_tokens,
        "generated_tokens": int(avg_tokens_gen),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "prefill_throughput_tps": prefill_throughput,
        "decode_throughput_tps": decode_throughput,
        "total_latency_ms": avg_total * 1000,
        "runs": num_runs,
        "load_time_s": load_time,
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
        "pp_mbu_roofline": round(pp_mbu, 4),
        "dec_mbu_roofline": round(dec_mbu, 4),
        "pp_mbu_measured": round(prefill_power.get('emc_bw_gb_s', 0) / peak_bw, 4) if peak_bw else 0,
        "dec_mbu_measured": round(decode_power.get('emc_bw_gb_s', 0) / peak_bw, 4) if peak_bw else 0,
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
            "gpu_used_mb": mem_decode['gpu_used'] if mem_decode else 0,
            "gpu_total_mb": mem_decode['gpu_total'] if mem_decode else 0,
        },
    }


def main():
    if len(sys.argv) < 5:
        print("Usage: bench_e2e.py <model_path> <prompt_tokens> <gen_tokens> <num_runs>")
        sys.exit(1)

    result = run_e2e_benchmark(
        model_path=sys.argv[1],
        prompt_tokens=int(sys.argv[2]),
        gen_tokens=int(sys.argv[3]),
        num_runs=int(sys.argv[4]),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
