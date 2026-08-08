#!/usr/bin/env python3
"""Power-instrumented TRT-Edge-LLM bench — produces the SAME JSON schema as the
other profiler_*/bench_e2e.py so it merges into the 62-col locked sweep with full
power/energy/MFU/MBU. Runs TRT-Edge llm_bench (--mode prefill / --mode decode)
as subprocesses and windows tegrastats power around each phase.

Usage: bench_e2e.py <engineDir> <pp> <gen> [runs]   (model fixed: Llama-3.2-1B)
Run inside trtedge container with -v tegrastats + -v /sys + this dir mounted.
"""
import sys, os, json, time, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from power_monitor import TegrastatsMonitor

ENGINE, PP, GEN = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
BIN = __import__("os").environ.get("TRT_BIN","/src/build/examples/llm/llm_bench")
# ---- Llama-3.2-1B-Instruct fixed constants ----
NUM_LAYERS, HID, HEADS, KV_HEADS, HEAD_DIM, FFN, VOCAB = 16, 2048, 32, 8, 64, 8192, 128256
NUM_PARAMS = 1_235_814_400
def _engine_bytes(d):
    try: return sum(__import__("os").path.getsize(__import__("os").path.join(d,f)) for f in __import__("os").listdir(d) if f.endswith((".engine",".safetensors")))
    except: return 0
PEAK_TFLOPS = 124.0                     # Thor fp16 dense
PEAK_BW = 273.1                         # GB/s
_n=ENGINE.lower()
_bpp = 0.5 if ('int4' in _n or 'nvfp4' in _n or 'awq' in _n) else (1.0 if ('int8' in _n or 'fp8' in _n) else 2.0)
WEIGHT_BYTES = int(NUM_PARAMS*_bpp)   # quant-aware weight traffic (fp16=2,int8/fp8=1,int4/nvfp4=0.5 bytes/param)
DEV = "AGX Thor 128GB"

def run_bench(args):
    out = subprocess.run([BIN, "--engineDir", ENGINE] + args, capture_output=True, text=True, timeout=3600).stdout
    out = re.sub(r'\x1b\[[0-9;]*m', '', out)
    return out

def grab(pat, s):
    m = re.search(pat, s); return float(m.group(1)) if m else 0.0

mon = TegrastatsMonitor(interval_ms=1)
with mon:
    time.sleep(2.0); idle = mon.get_power_breakdown()
    # --- prefill: repeat for a stable power window ---
    t0 = time.perf_counter_ns()
    pout = run_bench(["--mode", "prefill", "--inputLen", str(PP), "--useCudaGraph", "--noProfile", "--iterations", "40", "--warmup", "5"])
    t1 = time.perf_counter_ns()
    prefill = mon.get_power_breakdown(t0, t1)
    ttft_ms = grab(r"E2E Time[^0-9]*([0-9.]+)\s*ms", pout) or grab(r"Per-step avg:\s*([0-9.]+)", pout)
    # --- decode: real generation of GEN tokens from context PP ---
    t2 = time.perf_counter_ns()
    dout = run_bench(["--mode", "decode", "--pastKVLen", str(PP), "--osl", str(GEN), "--useCudaGraph", "--noProfile", "--iterations", "3", "--warmup", "2"])
    t3 = time.perf_counter_ns()
    decode = mon.get_power_breakdown(t2, t3)
    tpot_ms = grab(r"Per-step avg:\s*([0-9.]+)\s*ms", dout) or grab(r"E2E Time[^0-9]*([0-9.]+)\s*ms", dout)/max(GEN-1,1)

avg_prefill = ttft_ms/1000.0
avg_decode  = tpot_ms/1000.0 * GEN
prefill_tps = PP/avg_prefill if avg_prefill>0 else 0
decode_tps  = 1000.0/tpot_ms if tpot_ms>0 else 0
# energy (4-rail total * phase wall time)
prefill_energy = prefill.get('total4_mw',prefill.get('total_mw',0)) * avg_prefill
decode_energy  = decode.get('total4_mw',decode.get('total_mw',0))  * avg_decode
idle_gpu = idle.get('gpu_mw',0)
prefill_gpu_energy = max(prefill.get('gpu_mw',0)-idle_gpu,0)*avg_prefill
decode_gpu_energy  = max(decode.get('gpu_mw',0)-idle_gpu,0)*avg_decode
# flops / mfu
avg_ctx = PP + GEN/2.0
prefill_flops = 2*NUM_PARAMS*PP
decode_flops  = 2*NUM_PARAMS*GEN
pp_attn = 4*NUM_LAYERS*PP*PP*HEAD_DIM*KV_HEADS
dec_attn = 4*NUM_LAYERS*int(avg_ctx)*GEN*HEAD_DIM*KV_HEADS
prefill_tflops = prefill_flops/avg_prefill/1e12 if avg_prefill>0 else 0
decode_tflops  = decode_flops/avg_decode/1e12 if avg_decode>0 else 0
pp_mfu = prefill_tflops/PEAK_TFLOPS; dec_mfu = decode_tflops/PEAK_TFLOPS
# mbu (roofline + measured)
kv_bytes = 2*NUM_LAYERS*KV_HEADS*HEAD_DIM*2*int(avg_ctx)
est_pp_bw = WEIGHT_BYTES/avg_prefill/1e9 if avg_prefill>0 else 0
est_dec_bw = (WEIGHT_BYTES+kv_bytes)/avg_decode*GEN/1e9 if avg_decode>0 else (WEIGHT_BYTES+kv_bytes)*decode_tps/1e9
pp_mbu_roof = est_pp_bw/PEAK_BW; dec_mbu_roof = est_dec_bw/PEAK_BW
pp_mbu_meas = prefill.get('emc_bw_gb_s',0)/PEAK_BW; dec_mbu_meas = decode.get('emc_bw_gb_s',0)/PEAK_BW

print(json.dumps({
 "generated_tokens": GEN, "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
 "prefill_throughput_tps": prefill_tps, "decode_throughput_tps": decode_tps,
 "total_latency_ms": (avg_prefill+avg_decode)*1000, "memory_mb": 0, "peak_memory_mb": 0,
 "idle_power": idle, "prefill_power": prefill, "decode_power": decode,
 "prefill_energy_mj": prefill_energy, "decode_energy_mj": decode_energy,
 "prefill_gpu_energy_mj": prefill_gpu_energy, "decode_gpu_energy_mj": decode_gpu_energy,
 "num_params": NUM_PARAMS, "active_params": NUM_PARAMS,
 "prefill_tflops": round(prefill_tflops,4), "decode_tflops": round(decode_tflops,4),
 "prefill_attn_flops": pp_attn, "decode_attn_flops": dec_attn,
 "pp_mfu": round(pp_mfu,4), "dec_mfu": round(dec_mfu,4),
 "pp_mbu_measured": round(pp_mbu_meas,4), "dec_mbu_measured": round(dec_mbu_meas,4),
 "pp_mbu_roofline": round(pp_mbu_roof,4), "dec_mbu_roofline": round(dec_mbu_roof,4),
 "dec_mbu_roofline_total": round(dec_mbu_roof,4),
 "kv_cache_bytes_decode": kv_bytes, "peak_tflops_used": PEAK_TFLOPS,
 "peak_tflops_fp16_dense": PEAK_TFLOPS, "peak_bw_gb_s": PEAK_BW, "device_name": DEV,
 "active_weight_bytes": WEIGHT_BYTES,
}))
