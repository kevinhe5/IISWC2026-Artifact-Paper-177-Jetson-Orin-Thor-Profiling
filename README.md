# Artifact — IISWC 2026 "From Chat to Agents on the Edge"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21848837.svg)](https://doi.org/10.5281/zenodo.21848837)

All measured numbers used by the paper, organized as CSVs/JSON.
Each row is either a per-cell measurement or a per-condition aggregate;
no plotting is required to inspect the data.

## Two reproduction workflows

**Workflow A — plot-only, from shipped CSVs** (Available badge; ~ 5 min, any x86/ARM machine):

```bash
# regenerate any paper figure straight from the shipped data:
python3 scripts/plot/gen_fig11_pareto.py --out figs/
# or every figure at once:
for s in scripts/plot/gen_fig*.py; do python3 "$s" --out figs/; done
```

Every figure/table has a generator under `scripts/plot/` that reads only the
shipped CSV/JSON in `data/` (no embedded numbers). See `scripts/MANIFEST.md`
for the full script → data → figure map.

**Workflow B — full re-collect on real hardware** (Functional + Reproduced
badges; ~ 34 h on Orin, ~ 24 h on Thor):

Requires an AGX Orin 32 GB and/or AGX Thor 128 GB Developer Kit. Reproduces
the raw measurements from scratch, then Workflow A on top of the freshly
collected `sweep_locked.csv`.

```bash
# Orin (run from the repo root; same PROFILE_ROOT contract as Thor per AE §5)
export HF_TOKEN=hf_...                                       # Llama-3.2-1B is gated
export PROFILE_ROOT=/path/with/60GB/free                     # optional — defaults to <repo>/profile/
bash scripts/collect/orin/prepare_orin.sh                    # containers + models + engines
bash scripts/collect/orin/run_orin_collection.sh             # each stage resumable
for s in scripts/plot/gen_fig*.py; do python3 "$s" --out figs/; done

# Thor
export PROFILE_ROOT=/path/with/60GB/free
bash scripts/collect/thor/run_thor_collection.sh
```

Full step-by-step (prerequisites, timing per stage, verification, and
troubleshooting) is in **[WORKFLOW_B.md](WORKFLOW_B.md)**.

## Layout

```
artifact/
├── README.md
├── LICENSE
├── data/
│   ├── chat/                            ← §V single-turn microbenchmarks
│   │   ├── sweep_locked.csv             ← master sweep (fw × quant × pp × gen, Fig figA/figB/fits)
│   │   └── pytorch_compile.csv          ← torch.compile subset (figA supplement)
│   ├── agentic/                         ← §VI multi-turn SWE-bench-live
│   │   ├── llama_1B/                    ← Fig 12 dense (Llama-3.2-1B, 5 fw)
│   │   │   ├── vllm.csv
│   │   │   ├── sglang.csv
│   │   │   ├── trtllm.csv
│   │   │   ├── llamacpp.csv
│   │   │   └── pytorch.csv
│   │   ├── qwen3_4B/                    ← Fig 12 reasoning (Qwen3-4B, think-ON)
│   │   │   ├── vllm_thinkON.csv
│   │   │   ├── sglang_thinkON.csv
│   │   │   ├── sglang_thinkON_nostream.csv
│   │   │   ├── llamacpp_thinkON.csv
│   │   │   └── pytorch_thinkON.csv
│   │   └── radar/                       ← Fig 13 agentic-deployment radar
│   │       └── agentic_radar_axes.csv   ← 9 normalized axes × 5 frameworks
│   ├── nsys/                            ← §III kernel breakdown (Fig 6/7/8)
│   │   ├── kernel_categories.json       ← per-fw kernel category share (Fig 6/7 base)
│   │   ├── breakdown.json               ← wall-clock GPU/launch/host decomp (Fig 6)
│   │   ├── per_op.json                  ← per-operator timing detail
│   │   ├── all_overhead_summary.json    ← nsys profiler bias validation (Fig 6 bottom)
│   │   ├── nsys_overhead_{trtllm,vllm,llamacpp,pytorch}.json  ← per-fw overhead
│   │   ├── fig7_kernel_mix_quant.py     ← Fig 7 script (embedded DATA)
│   │   └── fig8_kernel_mix_longctx.py   ← Fig 8 script (embedded DATA)
│   ├── prefix_cache/                    ← §VI.A Table (prefix-cache TTFT stability)
│   │   ├── traces/                      ← 30-turn ReAct traces (raw)
│   │   │   └── {vllm,sglang,llamacpp}.log
│   │   └── scripts/
│   │       ├── gen_data.py              ← DATA[fw]["on"/"off"] embedded (per-turn TTFT)
│   │       └── gen_table.py             ← generates the paper's tab:prefix-cache
│   ├── bw_sweep/                        ← §B.2 sustained BW / MBU 1.37× update (STREAM-triad)
│   │   ├── triad_{orin,thor}_sustained.csv ← STREAM-triad at 3-GB footprint
│   │   ├── triad_sweep_{orin,thor}.csv  ← sweep across array sizes
│   │   └── size_dist_{orin,thor}.csv    ← per-array size sustained bandwidth
│   ├── mbu/                             ← Tab 3 MBU prediction (measured + predicted)
│   │   ├── measured_thor.csv
│   │   └── predictions_thor.csv
│   └── contention/                      ← Tab 5 co-tenant contention slowdown
│       └── summary_thor.csv
└── manifests/
    ├── framework_versions.csv           ← both devices: JetPack, CUDA, framework versions, backends, KV policies
    └── qwen3_chat_template.jinja        ← patched Qwen3 chat template used by llama.cpp Qwen3 runs
```

## Reproducibility notes

- **Sampling**: greedy across all throughput sweeps (temperature=0, top_k=1,
  argmax). Batch size 1 throughout.
- **Timing**: `time.perf_counter_ns()` with explicit `torch.cuda.synchronize()`.
  Agentic `phase_ms` uses client-side wall-clock `perf_counter`; prefill
  isolated by a `max_tokens=1` probe.
- **Power sampling**: `tegrastats --readall` for module rails + DRAM read
  from sysfs/hwmon (INA3221). Effective sampling rate ~74 Hz median.
- **Rails**: Orin exposes GPU / CPU / SOC / DRAM separately (4 rails);
  Thor exposes VDD_GPU + VDD_CPU_SOC_MSS (2 rails). Paper's `total4_mw` =
  sum of 4 Orin rails.
- **Agentic**: max_turns=30, max_tokens=2048 client cap, max-model-len=16384,
  vLLM `gpu_memory_utilization=0.55` for 1B/3B and 0.80 for Qwen3-4B,
  8-tool ReAct loop.
- **Model checkpoints**: pinned HuggingFace snapshots
  (Llama-3.2-1B-Instruct, Qwen3-4B).
- **CUDA graphs**: All reported vLLM cells (throughput and agentic) use
  `--enforce-eager` = graphs OFF; the V0/V1 A/B is the only place where
  graphs are explicitly toggled.
- **Attention backends per fw**: Orin vLLM = FLASH_ATTN + FlashInfer sampling;
  Orin SGLang = triton + `--disable-cuda-graph`; Orin llama.cpp = native
  flash-attn; Orin PyTorch = HF-default SDPA; Orin TRT-LLM = paged FMHA.
