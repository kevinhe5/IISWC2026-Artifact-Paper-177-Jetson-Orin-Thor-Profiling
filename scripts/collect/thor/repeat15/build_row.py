#!/usr/bin/env python3
# Builds one 62-col master-sweep CSV row from a bench_e2e.py JSON dict.
# Identical field order/formatting to sweep_thor_{vllm,sglang}.sh append_csv().
# Usage: build_row.py <framework> <quant> <pp> <gen> <ts>   (JSON on stdin)
import json, sys
fw, quant, pp, gen, ts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
d = json.load(sys.stdin)
ip = d.get('idle_power', {}); ppp = d.get('prefill_power', {}); dp = d.get('decode_power', {})
def g(o, k, dv=0):
    v = o.get(k, dv); return v if v is not None else dv
print(','.join(str(x) for x in [
 ts, fw, 'Llama-3.2-1B', quant, pp, gen, d.get('generated_tokens', gen),
 f"{d.get('ttft_ms',0):.2f}", f"{d.get('tpot_ms',0):.2f}",
 f"{d.get('prefill_throughput_tps',0):.1f}", f"{d.get('decode_throughput_tps',0):.1f}",
 f"{d.get('total_latency_ms',0):.2f}", f"{d.get('memory_mb',0):.0f}", f"{d.get('peak_memory_mb',0):.0f}",
 g(ip,'total_mw'), g(ip,'gpu_mw'), g(ip,'cpu_mw'), g(ip,'dram_mw'),
 g(ppp,'total_mw'), g(ppp,'gpu_mw'), g(ppp,'cpu_mw'), g(ppp,'soc_mw'), g(ppp,'dram_mw'),
 g(ppp,'gpu_util_pct'), g(ppp,'cpu_util_pct'), g(ppp,'emc_bw_gb_s'), int(bool(g(ppp,'samples_warning'))),
 g(dp,'total_mw'), g(dp,'gpu_mw'), g(dp,'cpu_mw'), g(dp,'soc_mw'), g(dp,'dram_mw'),
 g(dp,'gpu_util_pct'), g(dp,'cpu_util_pct'), g(dp,'emc_bw_gb_s'), g(dp,'gpu_temp_c'), int(bool(g(dp,'samples_warning'))),
 f"{d.get('prefill_energy_mj',0):.2f}", f"{d.get('decode_energy_mj',0):.2f}",
 f"{d.get('prefill_gpu_energy_mj',0):.2f}", f"{d.get('decode_gpu_energy_mj',0):.2f}",
 d.get('num_params',0), d.get('active_params', d.get('num_params',0)),
 d.get('prefill_tflops',0), d.get('decode_tflops',0), d.get('prefill_attn_flops',0), d.get('decode_attn_flops',0),
 d.get('pp_mfu',0), d.get('dec_mfu',0),
 d.get('pp_mbu_measured', d.get('pp_mbu',0)), d.get('dec_mbu_measured', d.get('dec_mbu',0)),
 d.get('pp_mbu_roofline', d.get('pp_mbu',0)), d.get('dec_mbu_roofline', d.get('dec_mbu',0)),
 d.get('kv_cache_bytes_decode',0), d.get('peak_tflops_used',0), d.get('peak_tflops_fp16_dense',0),
 d.get('peak_bw_gb_s',0), d.get('device_name',''), 0, 0,
 d.get('dec_mbu_roofline_total',0), d.get('active_weight_bytes',0),
]))
