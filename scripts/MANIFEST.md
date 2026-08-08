# Script Manifest — collection → data → figure

Maps every shipped Thor collection script to the data it produces and the paper figure/table it backs.
Fig/Tab numbers are PDF-verified (Overleaf HEAD 94f1f32).

## collect/harness/ — per-framework bench
| Script | Role |
|---|---|
| `profiler_{llamacpp,trtedge,trtllm}/bench_e2e.py` (shared) and `{orin,thor}/profiler_{vllm,sglang,pytorch}/bench_e2e.py` (device-specific) | one framework's E2E bench: prefill+decode latency, tegrastats power/energy, emits one 62-col JSON row per invocation (`... <model> <pp> <gen> <num_runs>`) |
| `power_monitor.py` | tegrastats rail sampling (~74 Hz) → per-phase mW/mJ |
| `gpu_utils.py` | clock/util helpers |

## collect/thor/sweep/ — master sweep (eager, 62-col)
| Script | Produces | Backs |
|---|---|---|
| `run_thor_full_sweep.sh` | master eager sweep grid (647 cells; base for the 15-run raw build → `data/chat/sweep_locked_thor.csv`) | Fig 2 roofline, Fig 3 MBU, Fig 5/6 fits, Fig 8 energy, Fig 11 pareto (fp16 pts), Tab 4 quant, footprint |
| `run_thor_quant_grid.sh` | quant-ladder rows in the sweep | Tab 4 quant-speedup, Fig 9 anchors |
| `benchmarks_thor_sweeps/sweep_thor_{vllm,sglang,pytorch,trtedge}.sh` | per-fw fp16 6×6 grid | as above |
| `benchmarks_thor_sweeps/sweep_thor_llamacpp{,_fa,_fused}.sh` | llama.cpp base + **fa (figures)** + fused | llama.cpp rows (fa backs the figures) |
| `benchmarks_thor_sweeps/sweep_thor_{vllm,trtedge,llamacpp}_quantgrid.sh` | quant grids | Tab 4, Fig 9 |

## collect/thor/repeat15/ — repeatability + graphs-ON + aggregation
| Script | Produces | Backs |
|---|---|---|
| `repeat_topup15.sh` → `repeat_grid.sh`, `repeat_grid_quant.sh`, `trtedge_repeat.sh`, `repeat_grid_trtedge_quant.sh` | `repeat_stats/*.jsonl` (15 fresh-load runs/cell) | §B.1.1 CV; input to the sweep build |
| `sweep_thor_cudagraph_15run.sh` (cells `cells_{vllm,sglang}_15run.txt`) + `build_row.py` | graphs-ON 15-run rows (vLLM/SGLang GGUF @128/128 + 2048/2048) → `thor_cudagraph_15run_rows.csv` | input to the sweep build (throughput cells) |
| `build_sweep_locked_thor_15run_raw.py` | folds the master grid + `repeat_stats` + graphs-ON rows into ONE `data/chat/sweep_locked_thor.csv` (RAW: ~15 rows/cell, graphs-ON included) | **figure-backing sweep** (loaders average per cell) |
| `refine_pareto_cudagraph_15run.py` | refines the 6 vLLM/SGLang gguf @128/128 rows of `pareto_thor_base.csv` to graphs-ON 15-run tps (pareto's own power pipeline kept) | Fig 11 |
| `agg_cv.py`, `agg_cv_quant.py` | CV stats (regenerated from the 15-run runs; not shipped as data) | §B.1.1 |

## collect/thor/nsys/ — kernel decomposition (Fig 4/9/10)
| Script | Produces | Backs |
|---|---|---|
| `nsys_capture.sh`, `longctx_capture{,2}.sh`, `trt_capture.sh` | `.nsys-rep` traces | raw for Fig 4/9/10 |
| `extract_kernel_categories_thor.py` | `data/nsys/kernel_categories_thor*.json` | Fig 9 kernel-mix-quant, Fig 10 longctx |
| `breakdown_thor.py` | `data/nsys/breakdown_thor.json` | Fig 4 nsys breakdown |
| `finalize_gen65536.py` | gen65536 long-ctx JSON | Fig 10 |

## collect/thor/agentic/ — SWE-bench-live (Fig 12)
| Script | Produces | Backs |
|---|---|---|
| `prepare_swebench_repos_thor.sh` | task repos | setup |
| `thor_swebench_{vllm,llamacpp,pytorch,sglang}.sh` | `data/agentic/*_thor/*.csv` | Fig 12 (Thor completeness; dense+thinkON figures are Orin-backed) |

## collect/thor/bw_sweep/ + mbu/
| Script | Produces | Backs |
|---|---|---|
| `bw_sweep/run_triad_sweep.sh` (+ `parse_torchprof_stream.py`) | `data/bw_sweep/triad_*_thor.csv` | §B.2 STREAM-triad |
| `mbu/mbu_rest_thor.sh`, `mbu_pytorch_thor.sh` | MBU measurement | Tab 3 mbu-prediction |

The figure generators live in `plot/` (one per figure, each reading only the shipped
CSV/JSON in `data/`). See `../README.md` for how to run them.
