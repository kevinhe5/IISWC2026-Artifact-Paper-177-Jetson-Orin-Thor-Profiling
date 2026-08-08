# Reproduction scripts — AGX Thor

All code used to **collect the data** and (under `plot/`) **generate the figures** for the Thor half
of the paper. Organized so an artifact reviewer can trace and re-run each experiment. The Orin half
ships the symmetric `collect/orin/` + its own plot scripts.

```
scripts/
├── collect/
│   ├── harness/                 # per-framework bench
│   │   ├── profiler_{llamacpp,trtedge,trtllm}/bench_e2e.py       # shared (device-agnostic)
│   │   ├── {orin,thor}/profiler_{vllm,sglang,pytorch}/bench_e2e.py  # device-specific
│   │   ├── power_monitor.py     # tegrastats rail sampling
│   │   └── gpu_utils.py
│   └── thor/
│       ├── sweep/               # master 62-col sweep (eager): backs most figures
│       ├── repeat15/            # N=15 repeatability (eager) + graphs-ON throughput + aggregators
│       ├── nsys/                # kernel capture + category/breakdown extractors (Fig 4/9/10)
│       ├── agentic/             # SWE-bench-live runners (Fig 12)
│       ├── bw_sweep/            # STREAM-triad sustained BW (§B.2)
│       └── mbu/                 # MBU measurement (Tab 3)
├── plot/                        # figure generators (one per figure) — read data/, write figs
└── MANIFEST.md                  # script → data file → figure/table map
```

## Environment (as measured)
- **Device:** Jetson AGX Thor 128 GB Dev Kit, JetPack R39.2 / JP7, CUDA 13.0 (containers), kernel
  6.8.12-tegra, GPU `sm_110` (Blackwell).
- **Clock lock:** `jetson_clocks`; every run verifies GPU `gpu-gpc-0` = 1575000000 Hz and EMC
  `bwmgr` = 4266000000 Hz. Between runs: `docker rm -f`, `drop_caches=3`, short settle.
- **Containers (jetson-containers image tags):**
  `thor:r38.3.arm64-sbsa-cu130-24.04-{vllm_0.12.0, sglang_0.5.7, llama_cpp, bitsandbytes, trtedge...}`.
  Framework versions are in `../manifests/framework_versions.csv`.
- **Models:** pinned HF snapshots (Llama-3.2-1B-Instruct, Qwen3-4B) + GGUF quant files; model
  weights are not shipped in the artifact (fetched from HF / quantized locally by the drivers).

**Reproducing elsewhere:** point the collection scripts at your data by setting **`PROFILE_ROOT`**
to the directory holding the Thor profile tree (`data/benchmarks/...`, `data/models/...`, and a
`work/` scratch subdir) — everything else is derived from it, no code edits needed. A few other
roots take the same override if your layout differs: `DATA_ROOT` (Orin `jetson-containers/data`),
`PAPER_JETSON`, `LOG_DIR`, `NSYS_TRACE_DIR`. Artifact outputs are written **relative to the repo**.
Container image tags and clock-lock frequencies live at the top of each driver. The per-framework
bench (`harness/`) is device-agnostic (it detects rails via `/sys/class/hwmon`).

## Reproduction order
**One entry point** runs all Thor collection + the sweep build in order (each stage independently resumable):
```bash
bash collect/thor/run_thor_collection.sh          # stages 1-3 + build
bash collect/thor/run_thor_collection.sh stage2   # or a single stage: stage1|stage2|stage3|build
```
1. **Master sweep (eager, 62-col):** `collect/thor/sweep/run_thor_full_sweep.sh` +
   `run_thor_quant_grid.sh` → the master grid. Each cell = `bench_e2e.py <model> <pp> <gen> 1`
   (3 warm-up + 1 measured, median over gen tokens).
2. **Repeatability N=15 (eager):** `collect/thor/repeat15/repeat_topup15.sh` (drives `repeat_grid*.sh`
   + `trtedge_repeat.sh`) → `repeat_stats/*.jsonl` (15 fresh-load runs/cell); `agg_cv.py` → CV stats (§B.1.1).
3. **Throughput graphs-ON (15-run):** `collect/thor/repeat15/sweep_thor_cudagraph_15run.sh` (cells in
   `cells_{vllm,sglang}_15run.txt`) → graphs-ON rows for the vLLM/SGLang GGUF throughput points →
   `thor_cudagraph_15run_rows.csv`. fp16 stays eager (see `../METHODOLOGY_NOTES.md`; Orin sm_87 runs these eager).
   - **Build:** `build_sweep_locked_thor_15run_raw.py` folds the master grid + `repeat_stats` + the graphs-ON
     rows into **one** `data/chat/sweep_locked_thor.csv` (raw, ~15 rows/cell; plot loaders average per cell —
     there is no separate cudagraph sweep). `refine_pareto_cudagraph_15run.py` updates the Fig 11 pareto input.
4. **nsys kernel decomposition:** `collect/thor/nsys/nsys_capture.sh` / `longctx_capture*.sh` /
   `trt_capture.sh` → `.nsys-rep`; `extract_kernel_categories_thor.py` + `breakdown_thor.py` +
   `finalize*.py` → `data/nsys/{kernel_categories_thor*,breakdown_thor}.json` (Fig 4/9/10).
5. **Agentic:** `collect/thor/agentic/prepare_swebench_repos_thor.sh` then
   `thor_swebench_{vllm,llamacpp,pytorch,sglang}.sh` → SWE-bench-live CSVs (Fig 12).
6. **BW / MBU:** `collect/thor/bw_sweep/run_triad_sweep.sh` (§B.2); `collect/thor/mbu/mbu_*_thor.sh` (Tab 3).

## Config choices (paper figures)
- Throughput figs: **graphs-ON** for vLLM+SGLang GGUF points, fp16 eager (see METHODOLOGY_NOTES).
- llama.cpp: **`llamacpp_fa`** (flash-attn) is the variant used for the figures.
- Figures backed by the **raw 15-run rows** shipped in `data/chat/sweep_locked_thor.csv`
  (loaders average per cell); the per-cell means match the single-run sweep within <0.6%.
- Contention (Tab 5): `data/contention/summary_thor.csv`.
