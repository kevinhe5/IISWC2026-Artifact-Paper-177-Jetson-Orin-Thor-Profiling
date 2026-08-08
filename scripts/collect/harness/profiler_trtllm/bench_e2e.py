#!/usr/bin/env python3
"""
End-to-end LLM benchmark using TensorRT-LLM: Prefill + Decode with power monitoring.
Outputs same CSV schema as llama.cpp profiler for cross-framework comparison.

Uses the TRT-LLM Python API (ModelRunnerCpp) for engine-based inference.
Models must be pre-built into TRT engines before benchmarking.
"""
import gc
import json
import os
import sys
import time

import torch
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

from power_monitor import TegrastatsMonitor
from gpu_utils import get_mem, drop_caches, print_memory_status, minimize_memory

# Shared device-aware spec (peak FP16 TFLOPS, peak BW, bus width) — lives at benchmarks/device_spec.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_spec import (  # noqa: E402
    get_spec, compute_flops, peak_tflops_for_quant,
    moe_active_params, kv_cache_bytes,
)


def get_engine_config(engine_dir: str) -> dict:
    """Read TRT-LLM engine config."""
    config_path = os.path.join(engine_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def get_num_params_from_config(config: dict) -> int:
    """Estimate parameter count from engine config."""
    build_cfg = config.get("build_config", {})
    pretrained = config.get("pretrained_config", {})
    hidden = pretrained.get("hidden_size", 0)
    layers = pretrained.get("num_hidden_layers", 0)
    n_heads = pretrained.get("num_attention_heads", 0)
    n_kv_heads = pretrained.get("num_key_value_heads", n_heads)
    intermediate = pretrained.get("intermediate_size", hidden * 4)
    vocab = pretrained.get("vocab_size", 32000)
    head_dim = hidden // n_heads if n_heads else 0

    attn = hidden * (n_heads + 2 * n_kv_heads) * head_dim + n_heads * head_dim * hidden
    ffn = 3 * hidden * intermediate
    emb = vocab * hidden * 2
    return layers * (attn + ffn) + emb


def get_arch_from_config(config: dict) -> dict:
    """Pull (num_layers, hidden_size, kv, MoE, quant) from TRT-LLM engine config."""
    pretrained = config.get("pretrained_config", {})
    build_cfg = config.get("build_config", {})
    n_heads = pretrained.get("num_attention_heads", 0)
    hidden = pretrained.get("hidden_size", 0)
    num_layers = pretrained.get("num_hidden_layers", 0)
    n_kv_heads = pretrained.get("num_key_value_heads", n_heads)
    head_dim = pretrained.get("head_dim", (hidden // n_heads) if n_heads else 0)
    intermediate = pretrained.get("intermediate_size", hidden * 4)
    moe_experts = pretrained.get("moe_num_experts", 0) or pretrained.get("num_local_experts", 0)
    moe_top_k = pretrained.get("moe_top_k", 0) or pretrained.get("num_experts_per_tok", 0)
    n_shared = pretrained.get("n_shared_experts", 0) or 0
    vocab = pretrained.get("vocab_size", 32000)
    # Engine quantization: e.g. "int4_weight_only" / "int8_weight_only" / "fp16" / "bf16".
    # The config keys may be present with None values (not just missing), so coerce via `or ''`.
    quant_cfg = pretrained.get("quantization", {}) or {}
    quant_algo = (quant_cfg.get("quant_algo") or "") if isinstance(quant_cfg, dict) else ""
    kv_quant = (quant_cfg.get("kv_cache_quant_algo") or "") if isinstance(quant_cfg, dict) else ""
    return {
        'num_layers': num_layers, 'hidden_size': hidden,
        'num_kv_heads': n_kv_heads, 'head_dim': head_dim,
        'intermediate_size': intermediate, 'vocab_size': vocab,
        'moe_experts': moe_experts, 'moe_top_k': moe_top_k, 'n_shared_experts': n_shared,
        'quant_algo': quant_algo.lower(), 'kv_quant': kv_quant.lower(),
    }


def _quant_name_from_config(config: dict) -> str:
    """Map TRT-LLM quant_algo → short name for peak_tflops_for_quant()."""
    pretrained = config.get("pretrained_config", {})
    quant_cfg = pretrained.get("quantization", {}) or {}
    q = (quant_cfg.get("quant_algo", "") if isinstance(quant_cfg, dict) else "") or ""
    q = q.lower()
    if 'int4' in q: return 'int4'
    if 'int8' in q: return 'int8'
    dtype = (pretrained.get('dtype') or '').lower()
    if dtype in ('bfloat16', 'bf16'): return 'bf16'
    if dtype in ('float16', 'fp16', 'half'): return 'fp16'
    return 'fp16'


def get_engine_size_mb(engine_dir: str) -> float:
    """Get total engine file size in MB."""
    total = 0
    for f in os.listdir(engine_dir):
        if f.endswith('.engine') or f.endswith('.plan'):
            total += os.path.getsize(os.path.join(engine_dir, f))
    return total / 1024 / 1024


def run_e2e_benchmark(engine_dir: str, tokenizer_dir: str,
                      prompt_tokens: int, gen_tokens: int, num_runs: int) -> dict:
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  TensorRT-LLM End-to-End Benchmark", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    config = get_engine_config(engine_dir)
    num_params = get_num_params_from_config(config)
    model_size_mb = get_engine_size_mb(engine_dir)
    model_size_gb = model_size_mb / 1024
    print(f"  Engine: {engine_dir}", file=sys.stderr)
    print(f"  Size: {model_size_mb:.0f} MB, Params: {num_params/1e6:.1f}M", file=sys.stderr)

    # Baseline
    gc.collect()
    torch.cuda.empty_cache()
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

    # Load engine
    print(f"\nLoading TRT-LLM engine...", file=sys.stderr)
    load_start = time.perf_counter()
    runner = ModelRunner.from_dir(engine_dir=engine_dir)
    load_time = time.perf_counter() - load_start
    print(f"  Loaded in {load_time:.2f}s", file=sys.stderr)

    gc.collect()
    time.sleep(0.5)
    mem_model = get_mem()
    print_memory_status("2. Engine loaded")

    # Build prompt tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    prompt = "Hello " * (prompt_tokens // 2 + 1)
    input_ids = tokenizer.encode(prompt)[:prompt_tokens]
    actual_prompt_tokens = len(input_ids)
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.int32)

    print(f"  Prompt tokens: {actual_prompt_tokens}, Gen tokens: {gen_tokens}", file=sys.stderr)

    # Warmup
    print("  Warmup (3 runs)...", file=sys.stderr)
    for _ in range(3):
        runner.generate(
            batch_input_ids=[input_ids_tensor],
            max_new_tokens=5,
            end_id=-1,  # Disable EOS stopping
            pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    print("  Warmup done.", file=sys.stderr)

    # Benchmark — two-call approach for accurate TTFT measurement
    # Call 1: generate 1 token → measures prefill (TTFT)
    # Call 2: generate all tokens → total time; decode = total - TTFT
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

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
        torch.cuda.empty_cache()

        # === PREFILL PHASE ===
        # Generate exactly 1 token to measure TTFT
        prefill_monitor = TegrastatsMonitor(interval_ms=1)
        with prefill_monitor:
            prefill_start_ns = time.perf_counter_ns()
            prefill_start = time.perf_counter()
            runner.generate(
                batch_input_ids=[input_ids_tensor],
                max_new_tokens=1,
                end_id=-1,  # Disable EOS stopping — generate exact token count
                pad_id=pad_id,
            )
            torch.cuda.synchronize()
            prefill_end = time.perf_counter()
            prefill_end_ns = time.perf_counter_ns()

        prefill_time = prefill_end - prefill_start

        # === FULL GENERATION (prefill + decode) ===
        decode_monitor = TegrastatsMonitor(interval_ms=1)
        with decode_monitor:
            decode_start_ns = time.perf_counter_ns()
            full_start = time.perf_counter()
            outputs = runner.generate(
                batch_input_ids=[input_ids_tensor],
                max_new_tokens=gen_tokens,
                end_id=-1,  # Disable EOS stopping — generate exact token count
                pad_id=pad_id,
            )
            torch.cuda.synchronize()
            full_end = time.perf_counter()
            decode_end_ns = time.perf_counter_ns()

        full_time = full_end - full_start
        # ModelRunner returns list of tensors: outputs[0][0] is the full sequence
        out = outputs[0][0]
        output_ids = out.cpu().tolist() if hasattr(out, 'cpu') else out
        generated_count = len(output_ids) - actual_prompt_tokens
        if generated_count < 0:
            generated_count = len(output_ids)  # Some versions don't include input
        decode_time = full_time - prefill_time
        if decode_time < 0:
            decode_time = full_time * 0.95

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
        print(f"    Run {i+1}: Total={total_time*1000:.1f}ms, {generated_count} tokens @ {decode_tps:.1f} tok/s", file=sys.stderr)

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
            # Semantic (device-normalized)
            'total_mw':  total_mw,
            'gpu_mw':    int(avg_bd(samples, 'gpu_mw')),
            'cpu_mw':    int(avg_bd(samples, 'cpu_mw')),
            'soc_mw':    int(avg_bd(samples, 'soc_mw')),
            'dram_mw':   dram_mw,
            'total4_mw': total_mw + dram_mw,
            # Legacy aliases (vdd_in_mw = total on both devices)
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
    decode_power  = build_power_dict(decode_power_samples)   # fixed: was dict(prefill_power) bug

    # Energy = average total power * phase time. Units: mW * s = mJ.
    # 4-rail total = 3 tegrastats rails + LPDDR5 cell rail (VDDQ_VDD2_1V8AO).
    prefill_energy = prefill_power['total4_mw'] * avg_prefill
    decode_energy  = decode_power['total4_mw']  * avg_decode
    # GPU-attributable energy = (gpu_mw - idle_gpu_mw) * t
    idle_gpu_mw = idle_power.get('gpu_mw', 0)
    prefill_gpu_energy = max(prefill_power['gpu_mw'] - idle_gpu_mw, 0) * avg_prefill
    decode_gpu_energy  = max(decode_power['gpu_mw']  - idle_gpu_mw, 0) * avg_decode

    spec = get_spec()
    emc_freq = prefill_power.get('emc_freq_mhz', spec['emc_freq_max_mhz'])
    peak_bw = prefill_power.get('emc_peak_bw_gb_s') or emc_freq * spec['bus_bytes'] * 2 / 1000

    # FLOPs with attention quadratic term (MLPerf / NVIDIA NeMo convention)
    arch = get_arch_from_config(config)
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
    # MFU against peak at the actual compute precision the engine uses.
    quant_name = _quant_name_from_config(config)
    peak_tflops_used = peak_tflops_for_quant(quant_name)
    pp_mfu = (prefill_tflops / peak_tflops_used) if peak_tflops_used else 0
    dec_mfu = (decode_tflops / peak_tflops_used) if peak_tflops_used else 0

    # KV-aware bandwidth roofline: decode traffic = weights + KV per token.
    kv_bytes_decode = kv_cache_bytes(
        num_layers=arch['num_layers'], num_kv_heads=arch['num_kv_heads'],
        head_dim=arch['head_dim'], seq_len=int(avg_ctx_decode), batch_size=1, dtype_bytes=2,
    )
    model_bytes = model_size_gb * 1e9
    est_prefill_bw = (model_bytes / avg_prefill / 1e9) if avg_prefill > 0 else 0
    est_decode_bw  = ((model_bytes + kv_bytes_decode) * avg_decode_tokens / avg_decode / 1e9) if avg_decode > 0 else 0
    pp_mbu = (est_prefill_bw / peak_bw) if peak_bw else 0
    dec_mbu = (est_decode_bw / peak_bw) if peak_bw else 0

    # Cleanup
    del runner
    gc.collect()
    torch.cuda.empty_cache()
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
            "gpu_used_mb": mem_decode['gpu_used'] if mem_decode else 0,
            "gpu_total_mb": mem_decode['gpu_total'] if mem_decode else 0,
        },
    }


def main():
    if len(sys.argv) < 6:
        print("Usage: bench_e2e.py <engine_dir> <tokenizer_dir> <prompt_tokens> <gen_tokens> <num_runs>")
        sys.exit(1)

    result = run_e2e_benchmark(
        engine_dir=sys.argv[1],
        tokenizer_dir=sys.argv[2],
        prompt_tokens=int(sys.argv[3]),
        gen_tokens=int(sys.argv[4]),
        num_runs=int(sys.argv[5]),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
