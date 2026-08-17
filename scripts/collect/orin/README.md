# Orin collection scripts (Workflow B)

Orin-side collection drivers for the IISWC 2026 "From Chat to
Agents on the Edge" artifact.

```bash
cd <repo-root>
export HF_TOKEN=hf_...                    # Llama-3.2-1B-Instruct is gated
export PROFILE_ROOT=/path/with/60GB/free  # optional — defaults to <repo>/profile/

bash scripts/collect/orin/prepare_orin.sh           # ~ 2-3 h
bash scripts/collect/orin/run_orin_collection.sh    # ~ 34 h
```

(`prepare_orin.sh` also installs the bench harness into
`$PROFILE_ROOT/benchmarks/` via `install_harness.sh` — no manual copy.)

Then re-render the paper's figures against the freshly collected
`data/chat/sweep_locked.csv`:
```bash
for s in scripts/plot/gen_fig*.py; do python3 "$s" --out figs/; done
```

Full walkthrough (prerequisites, per-stage timing, verification,
troubleshooting) is in the repo-root [`WORKFLOW_B.md`](../../../WORKFLOW_B.md).

### Entry points

**Reviewers run exactly two scripts**: `prepare_orin.sh` once, then
`run_orin_collection.sh`. The second one drives everything below it and
produces every Orin-side file the plotting scripts read.

| script                          | role                                                                       |
|---------------------------------|----------------------------------------------------------------------------|
| `grant_sudo.sh`                 | one-time (as root): installs a sudoers whitelist for the 5 privileged commands the pipeline needs |
| `prepare_orin.sh`               | one-shot: harness install, docker pulls, container builds, HF snapshots, GGUFs, TRT engines |
| `install_harness.sh`            | assembles `harness/` into the runnable `$PROFILE_ROOT/benchmarks/` layout (called by prepare) |
| `run_orin_collection.sh`        | **the** collection driver: preflight → sweep → side grids → repeat → nsys → build |
| `preflight.sh`                  | swap / swappiness / jetson_clocks / drop_caches (called by the driver)     |
| `side_sweeps.sh`                | compile / flash-attn / long-context grids (called by the driver's `side` stage) |
| `build_sweep_locked_orin.py`    | folds raw sweeps into the `data/chat/*.csv` files the plots read (called by the driver's `build` stage) |
| `dockerfiles/Dockerfile.*`      | reference recipes for `sglang-orin` and `bitsandbytes-bench` images        |

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
│   ├── sweep/                    sweep.sh — THE sweep worker (SWEEP_SCOPE=full
│   │                             for stage 1, SWEEP_SCOPE=1b for stage-2 reps)
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
- Data root at `$PROFILE_ROOT` (models, sweep outputs, HF cache;
  defaults to `<repo>/profile/` when unset)

## Per-directory contents

- **sweep/** — `sweep.sh`, the single sweep worker for both measurement
  stages: `SWEEP_SCOPE=full` (stage 1: 5 fw incl. SGLang × 3 models ×
  quants × 6×6 pp/gen grid) or `SWEEP_SCOPE=1b` (stage-2 reps: same minus
  the 8B/Mixtral blocks). Unifies the two historical worker scripts.
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
