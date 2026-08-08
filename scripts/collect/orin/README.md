# Orin collection scripts

Orin-side collection drivers for the IISWC 2026 "From Chat to
Agents on the Edge" artifact.

## Layout

```
scripts/collect/
├── harness/                          ← per Thor's finalized layout (msg #35)
│   ├── power_monitor.py              ← SINGLE shared (Orin+Thor rail-aware)
│   ├── gpu_utils.py                  ← SINGLE shared (identical both platforms)
│   ├── profiler_llamacpp/bench_e2e.py ← SINGLE shared (identical both platforms)
│   ├── profiler_trtllm/bench_e2e.py   ← Orin engine
│   ├── profiler_trtedge/bench_e2e.py  ← Thor engine (added by Thor session)
│   ├── orin/profiler_{vllm,sglang,pytorch}/bench_e2e.py  ← Orin-specific vLLM/SGLang/PyTorch
│   ├── thor/profiler_{vllm,sglang,pytorch}/bench_e2e.py  ← Thor-specific (added by Thor)
│   └── harness_delta/orin/profiler_pytorch/gpu_utils.py  ← Orin-only extended pytorch helpers
├── orin/               ← Orin-specific drivers (this dir)
│   ├── sweep/                    single-run 6×6×fw×quant×model main sweep
│   ├── repeat15/                 N=15 repeatability drivers (full sweep re-runs)
│   ├── nsys/                     Nsight Systems profile capture + JSON extractors
│   ├── agentic/                  SWE-bench-live runners (per framework)
│   └── bw_sweep/                 STREAM-triad bandwidth sweep (CUDA source + driver)
└── thor/               ← Thor-specific drivers (owned by Thor session)
```

## Container tags (Orin)

Referenced by scripts under `orin/`:

| Framework    | Image                                              |
|--------------|----------------------------------------------------|
| vLLM         | `dustynv/vllm:0.8.6-r36.4-cu128-24.04`             |
| SGLang       | `sglang-orin:0.4.6-sm87`                           |
| llama.cpp    | `dustynv/llama_cpp:b5283-r36.4-cu128-24.04`        |
| TensorRT-LLM | `dustynv/tensorrt_llm:0.12-r36.4.0`                |
| PyTorch (HF) | `bitsandbytes-bench:r36.4.0`                       |

## Run environment invariants

All Orin scripts assume:

- `jetson_clocks` locked (GPU 1.3 GHz, EMC 3.199 GHz, MAXN)
- Only one Docker container running at a time (concurrent GPU allocation is
  not supported by the harness)
- 32 GB unified memory total (constrains Mixtral cell — see
  `sweep/sweep.sh` gpu_layers=16 note)
- Data root at `/nvme/ispass/jetson-containers/data` (models, sweep
  outputs, HF cache)

## Per-directory contents

- **sweep/** — `sweep.sh` (main 861-cell 5-fw × 3-model × 4/10 quants ×
  6×6 pp/gen grid), `run_locked_sweep.sh` (clock-locked wrapper),
  `resume_locked_sweep.sh` (mid-run resumer using row-exists skip).
- **repeat15/** — `sweep_1B_only.sh` (per-rep sweep restricted to
  Llama-3.2-1B for the N=15 repeatability data), `run_reps_4_to_14.sh`
  (extends reps 3 → 14), `run_rep15.sh` (final rep15 wrapper),
  `run_n10_sweep.sh` (per-cell N=10 iteration protocol for the Q1
  headline anchor cells).
- **nsys/** — `run_profile.sh` / `run_baseline.sh` / `run_profile_repeat.sh`
  (nsys capture drivers); `extract_breakdown.py` /
  `extract_kernel_categories.py` / `extract_per_op.py` /
  `launch_gap_dist.py` (post-processing extractors producing the JSON
  files under `data/nsys/`).
- **agentic/** — `rerun_swebench_live_{vllm,sglang,trtllm,llamacpp,pytorch}.sh`
  (5-framework SWE-bench-live runners producing the per-fw CSVs in
  `data/agentic/`).
- **bw_sweep/** — `run_triad_sweep.sh` + `triad_bw_sweep.cu` /
  `triad_bw_sweep_dist.cu` (STREAM-triad bandwidth micro-benchmark for
  the MBU prediction analysis).
## Harness — shared copy

`scripts/collect/harness/` contains the per-framework bench harness that
BOTH Orin and Thor use. The harness reads platform via `/sys/class/hwmon`
labels at runtime (Orin exposes 4 rails including DRAM; Thor exposes 2
combined rails), so a single copy serves both platforms.

**Divergences to flag** (if any surface between Orin & Thor copies):
- `bench_e2e.py`: `num_runs` and warmup-count defaults (per-fw)
- `power_monitor.py`: rail-label mapping (Orin vs Thor differ but the code
  path is defensive — searches for known labels)
- `gpu_utils.py`: model-size / KV-size computations (model-agnostic)

If Thor's copy diverges from what's checked in here, we reconcile before
the artifact ships. Contact channel: handoff MCP.
