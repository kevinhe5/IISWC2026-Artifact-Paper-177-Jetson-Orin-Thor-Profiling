# Shared bench harness

Per-framework end-to-end bench used by both platforms' collection drivers.

- `gpu_utils.py`, `power_monitor.py`, `profiler_llamacpp/bench_e2e.py` — **single shared copy**
  (byte-identical across platforms; `power_monitor.py` is platform-aware: auto-detects Orin's 4 rails
  and Thor's 2-rail VDD_CPU_SOC_MSS layout).
- `profiler_trtedge/` (Thor TensorRT-Edge) and `profiler_trtllm/` (Orin TensorRT-LLM) — different engines.
- `thor/profiler_{vllm,sglang,pytorch}/` and `orin/profiler_{vllm,sglang,pytorch}/` — **platform-specific
  variants** (differ in env knobs: e.g. Thor vLLM uses VLLM_ASYNC_SCHEDULING + SGLANG_DISABLE_CUDA_GRAPH
  for the graphs-ON throughput cells; Orin vLLM uses VLLM_MAX_NUM_SEQS/SWAP_SPACE for its memory budget).
  Each platform's drivers call its own copy so reproduction is faithful.
