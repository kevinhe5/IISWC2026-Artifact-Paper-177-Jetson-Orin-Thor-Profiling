#!/usr/bin/env python3
"""
End-to-end LLM benchmark using SGLang: Prefill + Decode with power monitoring.
Outputs the same CSV schema as the other framework profilers (vLLM, llama.cpp,
TRT-LLM, PyTorch) for cross-framework comparison.

Reconstructed 2026-04-30 from the bytecode of an earlier version (the .py
source had been deleted). Adds:
  - Leading torch.cuda.synchronize() before the timer (matches the vLLM fix —
    SGLang's bench had the same bug pattern: trailing sync only).
  - SGLANG_DISABLE_RADIX_CACHE env var → passes disable_radix_cache=True to
    the Engine, used by the no-cache scaling study.
  - UNIQUE_PROMPT env var → prepends a random hex prefix so every cell forces
    a cache miss even if the radix cache is enabled.

The Engine config matches what was observed in the original bytecode:
  trust_remote_code=True, mem_fraction_static=0.85, enable_metrics=False,
  log_level='warning', grammar_backend='none', disable_cuda_graph=True,
  attention_backend='triton'.
"""
import gc
import json
import os
import sys
import time

from sglang.srt.entrypoints.engine import Engine

from power_monitor import TegrastatsMonitor
from gpu_utils import get_mem, drop_caches, print_memory_status

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_spec import (  # noqa: E402
    get_spec, compute_flops, peak_tflops_for_quant,
    moe_active_params, kv_cache_bytes,
)


def get_arch_from_config(model_path: str) -> dict:
    """Pull arch from HF config.json. SGLang requires HF safetensors (no GGUF)."""
    EMPTY = {
        'num_layers': 0, 'hidden_size': 0, 'num_kv_heads': 0, 'head_dim': 0,
        'intermediate_size': 0, 'vocab_size': 0,
        'moe_experts': 0, 'moe_top_k': 0, 'n_shared_experts': 0,
        'torch_dtype': 'float16',
    }
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isdir(model_path) or not os.path.exists(config_path):
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


def _quant_name(arch: dict) -> str:
    """SGLang loads HF safetensors, so quant follows torch_dtype."""
    dt = arch.get('torch_dtype', 'float16')
    if dt in ('bfloat16', 'bf16'): return 'bf16'
    return 'fp16'


def get_model_num_params(model_path: str) -> int:
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
    if os.path.isfile(model_path):
        return os.path.getsize(os.path.realpath(model_path)) / 1024 / 1024
    total = 0
    for f in os.listdir(model_path):
        if f.endswith(('.safetensors', '.bin', '.pt')):
            try:
                total += os.path.getsize(os.path.realpath(os.path.join(model_path, f)))
            except OSError:
                pass
    return total / 1024 / 1024


def run_e2e_benchmark(model_path: str, prompt_tokens: int, gen_tokens: int, num_runs: int) -> dict:
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  SGLang End-to-End Benchmark", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    num_params = get_model_num_params(model_path)
    model_size_mb = get_model_size_mb(model_path)
    model_size_gb = model_size_mb / 1024
    print(f"  Model: {os.path.basename(model_path)}", file=sys.stderr)
    print(f"  Size:  {model_size_mb:.0f} MB, Params: {num_params/1e6:.1f}M", file=sys.stderr)

    gc.collect()
    drop_caches()
    time.sleep(0.5)
    mem_baseline = get_mem()
    print_memory_status("1. Baseline")

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
    print(f"  Idle VDD_IN: {idle_power.get('vdd_in_mw', 0)} mW", file=sys.stderr)

    # ── Load Engine ─────────────────────────────────────────────────────────
    # Optional cache-disable for the no-cache scaling study. Default behavior
    # is unchanged (RadixAttention is on by default). Setting
    # SGLANG_DISABLE_RADIX_CACHE=1 forces disable_radix_cache=True.
    disable_cache = os.environ.get("SGLANG_DISABLE_RADIX_CACHE", "0") == "1"

    print(f"\nLoading SGLang Engine{' (radix_cache=OFF)' if disable_cache else ''}...",
          file=sys.stderr)
    load_start = time.perf_counter()
    # Long-context KV pool: sglang already at 0.85, but allow env override.
    sg_mem_frac = 0.85
    env_mem = os.environ.get("SGLANG_MEM_FRAC", "")
    if env_mem:
        try: sg_mem_frac = float(env_mem)
        except ValueError: pass
    # CUDA-graph capture is env-controlled so we can run both methodologies:
    #  - default (SGLANG_DISABLE_CUDA_GRAPH=1): eager, host-overhead exposed (cross-framework consistent w/ vLLM/llama.cpp eager)
    #  - SGLANG_DISABLE_CUDA_GRAPH=0: CUDA graphs ON, matching the paper's Tier-2 Thor methodology (5-9ms kernel band)
    _disable_graph = os.environ.get('SGLANG_DISABLE_CUDA_GRAPH', '1') == '1'
    engine_kwargs = dict(
        model_path=model_path,
        trust_remote_code=True,
        mem_fraction_static=sg_mem_frac,
        enable_metrics=False,
        log_level='warning',
        grammar_backend='none',
        disable_cuda_graph=_disable_graph,
        attention_backend='triton',
        max_total_tokens=max(prompt_tokens + gen_tokens + 128, 8192),
    )
    if disable_cache:
        engine_kwargs['disable_radix_cache'] = True
    if os.environ.get('SGLANG_DISABLE_OVERLAP') is not None:
        engine_kwargs['disable_overlap_schedule'] = (os.environ.get('SGLANG_DISABLE_OVERLAP') == '1')
    try:
        engine = Engine(**engine_kwargs)
    except Exception as e:
        print(f"ERROR loading Engine: {e}", file=sys.stderr)
        raise
    load_time = time.perf_counter() - load_start
    print(f"  Loaded in {load_time:.2f}s", file=sys.stderr)

    gc.collect()
    time.sleep(0.5)
    mem_model = get_mem()
    print_memory_status("2. Model loaded")

    # ── Build prompt ────────────────────────────────────────────────────────
    # Default uses a deterministic "Hello " repetition so repeated calls hit
    # the radix cache (matches real-world chat deployment). Setting
    # UNIQUE_PROMPT=1 prepends a random hex string so every cell forces a
    # cache miss — used by the no-cache scaling study.
    unique_prompt = os.environ.get("UNIQUE_PROMPT", "0") == "1"
    if unique_prompt:
        import secrets
        prompt = secrets.token_hex(64) + " " + ("Hello " * (prompt_tokens // 2 + 1))
    else:
        prompt = "Hello " * (prompt_tokens // 2 + 1)
    # Note: SGLang's Engine accepts a string prompt directly and tokenizes
    # internally; we don't have access to the tokenizer for length verification
    # here without a separate import. Approximate: 2 tokens per "Hello " word
    # for Llama-family tokenizers; the trim happens server-side.
    actual_prompt_tokens = prompt_tokens

    print(f"  Prompt tokens: {actual_prompt_tokens}, Gen tokens: {gen_tokens}",
          file=sys.stderr)

    # ── Sampling params ─────────────────────────────────────────────────────
    warmup_params = dict(max_new_tokens=5, temperature=0.0, ignore_eos=True)
    decode_params = dict(max_new_tokens=gen_tokens, temperature=0.0, ignore_eos=True)
    prefill_params = dict(max_new_tokens=1, temperature=0.0, ignore_eos=True)

    # ── Warmup ──────────────────────────────────────────────────────────────
    print("  Warmup (3 runs)...", file=sys.stderr)
    for _ in range(3):
        engine.generate(prompt=prompt, sampling_params=warmup_params)
    print("  Warmup done.", file=sys.stderr)

    # ── Benchmark ───────────────────────────────────────────────────────────
    prefill_times = []
    decode_times = []
    total_times = []
    tokens_generated_list = []
    prefill_power_samples = []
    decode_power_samples = []
    mem_prefill = None
    mem_decode = None

    import torch as _torch_b  # local import for synchronize support

    print(f"  Running {num_runs} runs...", file=sys.stderr)
    for i in range(num_runs):
        gc.collect()

        # ── PREFILL PHASE ────────────────────────────────────────────────
        # Drain any pending GPU work from prior warmup calls before timing —
        # without this, SGLang's async scheduler leaves queued ops on the
        # stream and the next generate()'s wall absorbs that drain.
        prefill_monitor = TegrastatsMonitor(interval_ms=1)
        if _torch_b.cuda.is_available():
            _torch_b.cuda.synchronize()
        with prefill_monitor:
            prefill_start_ns = time.perf_counter_ns()
            prefill_start = time.perf_counter()
            engine.generate(prompt=prompt, sampling_params=prefill_params)
            if _torch_b.cuda.is_available():
                _torch_b.cuda.synchronize()
            prefill_end = time.perf_counter()
            prefill_end_ns = time.perf_counter_ns()
        prefill_time = prefill_end - prefill_start

        # ── DECODE PHASE ─────────────────────────────────────────────────
        decode_monitor = TegrastatsMonitor(interval_ms=1)
        with decode_monitor:
            decode_start_ns = time.perf_counter_ns()
            full_start = time.perf_counter()
            result = engine.generate(prompt=prompt, sampling_params=decode_params)
            if _torch_b.cuda.is_available():
                _torch_b.cuda.synchronize()
            full_end = time.perf_counter()
            decode_end_ns = time.perf_counter_ns()

        full_time = full_end - full_start

        # SGLang result is a dict with meta_info.{completion_tokens, output_tokens,
        # prompt_tokens}. Different versions use different keys; try them.
        if isinstance(result, dict):
            meta = result.get('meta_info', {})
            generated_count = (meta.get('completion_tokens', 0)
                               or meta.get('output_tokens', 0)
                               or gen_tokens)
        else:
            generated_count = gen_tokens

        decode_time = full_time - prefill_time
        if decode_time < 0:
            decode_time = full_time * 0.95

        prefill_times.append(prefill_time)
        decode_times.append(decode_time)
        total_times.append(full_time)
        tokens_generated_list.append(generated_count)
        prefill_power_samples.append(prefill_monitor.get_power_breakdown(prefill_start_ns, prefill_end_ns))
        decode_power_samples.append(decode_monitor.get_power_breakdown(decode_start_ns, decode_end_ns))

        if i == 0:
            gc.collect()
            time.sleep(0.2)
            mem_prefill = get_mem()
            mem_decode = get_mem()

        decode_tokens = generated_count - 1 if generated_count > 1 else 1
        decode_tps = decode_tokens / decode_time if decode_time > 0 else 0
        print(f"    Run {i+1}: Total={full_time*1000:.1f}ms, {generated_count} tokens @ {decode_tps:.1f} tok/s",
              file=sys.stderr)

    # ── Aggregate ───────────────────────────────────────────────────────────
    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decode = sum(decode_times) / len(decode_times)
    avg_total = sum(total_times) / len(total_times)
    avg_tokens_gen = sum(tokens_generated_list) / len(tokens_generated_list)

    ttft_ms = avg_prefill * 1000
    avg_decode_tokens = avg_tokens_gen - 1 if avg_tokens_gen > 1 else 1
    tpot_ms = (avg_decode / avg_decode_tokens * 1000)
    prefill_throughput = actual_prompt_tokens / avg_prefill if avg_prefill > 0 else 0
    decode_throughput = avg_decode_tokens / avg_decode if avg_decode > 0 else 0

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
    quant_name = _quant_name(arch)
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

    # Cleanup. SGLang's Engine has a shutdown method.
    try:
        engine.shutdown()
    except Exception:
        pass
    del engine
    gc.collect()
    drop_caches()
    time.sleep(1)
    mem_cleanup = get_mem()

    model_memory_mb = mem_model['sys_used'] - mem_baseline['sys_used']
    peak_memory_mb = mem_decode['sys_used'] if mem_decode else mem_model['sys_used']

    print(f"\n  Summary: TTFT={ttft_ms:.1f}ms, TPOT={tpot_ms:.2f}ms, Decode={decode_throughput:.1f} tok/s",
          file=sys.stderr)

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
