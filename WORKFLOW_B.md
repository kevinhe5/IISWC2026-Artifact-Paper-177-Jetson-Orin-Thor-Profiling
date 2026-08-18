# Workflow B — end-to-end re-collection on real hardware

This document is the AE reviewer walkthrough for **Workflow B** (re-collect all
measurements from scratch on a Jetson device, then regenerate every figure
from the freshly collected data). Workflow A (plot-only, off the shipped
CSVs) is documented in the top-level `README.md`.

Two independent execution paths are given, one per device:

- [Orin (AGX Orin 32 GB)](#orin-agx-orin-32-gb)
- [Thor (AGX Thor 128 GB)](#thor-agx-thor-128-gb)

Each path is **one command** for setup, **one command** for collection,
**one command** to regenerate every figure — plus a verification step that
compares the newly built CSV against the shipped one.

---

## Orin (AGX Orin 32 GB)

### 0. Prerequisites

| item             | value                                                         |
|------------------|---------------------------------------------------------------|
| Device           | Jetson AGX Orin 32 GB Developer Kit                           |
| GPU arch         | sm_87 (Ampere)                                                |
| JetPack          | 6.2 GA (L4T r36.4)                                            |
| Kernel           | 5.15.148-tegra                                                |
| CUDA runtime     | 12.6 host / containers ship 12.8                              |
| Docker           | ≥ 24.0, `nvidia` container runtime configured                 |
| Free disk        | ~ 60 GB on `$PROFILE_ROOT` (weights + engines + sweep outputs; defaults to `<repo>/profile/`) |
| sudo             | PASSWORDLESS sudo required for the measurement stages — see below |
| HF_TOKEN         | required (Llama-3.2-1B-Instruct is gated)                     |
| Wall-clock       | ~ 34 h for the full default pipeline                          |

**sudo requirements.** The measurement stages need a handful of root
operations: `jetson_clocks` (lock clocks), `nvpmodel` (MAXN power mode),
`sysctl vm.swappiness`, `echo 3 > /proc/sys/vm/drop_caches` (page-cache
eviction between runs — on unified-memory Jetson an undropped cache can
starve GPU allocations), and a debugfs read to verify the locked EMC
frequency. Preflight refuses to start a measurement stage when it can't
perform or verify these. **One-time setup (recommended):**

```bash
sudo bash scripts/collect/orin/grant_sudo.sh
```

This installs `/etc/sudoers.d/orin-artifact`, whitelisting passwordless
sudo for exactly those five commands (nothing broader; the file is
`visudo`-validated before install). Undo afterwards with
`sudo rm /etc/sudoers.d/orin-artifact`.

Preflight tries each privileged command individually, so partial/site
whitelists are honored too. It hard-fails ONLY on measurement-critical,
verifiable conditions: GPU/EMC clocks not locked, power mode not MAXN,
excess swap, or missing data layout. The hygiene extras (cache drop,
swappiness, EMC verification) degrade to loud warnings and the run
proceeds. Minimum root need if you skip grant_sudo.sh entirely: have an
administrator run `sudo jetson_clocks` once per boot (plus
`sudo nvpmodel -m 0` if the mode isn't MAXN).

*Alternative:* run the collection itself under `sudo -E` — but then
everything under `PROFILE_ROOT` and the HF cache in your home directory
end up root-owned, which breaks later non-root runs; prefer the
whitelist. The sweeps print a loud warning (not a silent skip) whenever
cache-dropping fails mid-run. The `build` stage is pure CSV folding and
runs without sudo or a GPU.

Verify hardware + JetPack:
```bash
cat /etc/nv_tegra_release            # expected: R36 (release), REVISION: 4.x
nvidia-smi -q -d COMPUTE | head -5   # expected: sm_87
docker info | grep -i runtime        # expected: Default Runtime: nvidia
```

### 1. Configure environment

All commands below are run **from the repo root** (`cd` into your clone
first). Only one export is required:

```bash
cd IISWC2026-Artifact-Paper-177-Jetson-Orin-Thor-Profiling
export HF_TOKEN=hf_...                                    # REQUIRED — see below
```

The ~60 GB of models, engines, and sweep outputs go under `PROFILE_ROOT`
(same variable Thor uses; §5-7 of the AE appendix). **If unset it defaults
to `<repo>/profile/`** (gitignored), which is fine when your clone lives on
a large filesystem. If the clone sits on a small disk (typical Jetson eMMC),
point it at your NVMe instead:

```bash
export PROFILE_ROOT=/path/on/big/disk   # optional — needs 60+ GB free
```

The scripts create it and every subdirectory themselves; the disk preflight
aborts with a clear message if the chosen filesystem is too small.

The framework bench-runners (harness) are installed into
`$PROFILE_ROOT/benchmarks/` **automatically by `prepare_orin.sh`** (step 0b —
it assembles the repo's deduplicated `scripts/collect/harness/` into the
flattened per-profiler layout the containers mount at `/benchmarks`). No
manual copy is needed; to re-install by hand, run
`bash scripts/collect/orin/install_harness.sh`. Do **not**
`cp -a` the harness directory as-is — the raw repo layout is not runnable.

### 2. One-shot environment setup — `prepare_orin.sh`

> **You MUST export `HF_TOKEN` first.** `meta-llama/Llama-3.2-1B-Instruct`
> is a gated model. Get a read token at
> https://huggingface.co/settings/tokens and accept the licence at
> https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct.
> Without the token the script exits at step 3/5 with `[FATAL] HF_TOKEN
> is not set.` — no engines will be built.

> The script also refuses to start with < 60 GB free on `PROFILE_ROOT`
> (or < 25 GB on the docker root), exit 5 — an ENOSPC mid-run leaves
> partial engines that the resume logic would wrongly treat as complete.

```bash
bash scripts/collect/orin/prepare_orin.sh
```

What it does (each step is skipped if the output already exists):

| step | action                                                  | wall-clock |
|------|---------------------------------------------------------|------------|
| 1    | `docker pull` for the 3 upstream framework containers   | 20-40 min  |
| 2    | `docker build` for `sglang-orin` + `bitsandbytes-bench` | 20-30 min  |
| 3    | HF snapshot download (Llama-3.2-1B, Qwen3-4B)           | 5-15 min   |
| 4    | GGUF quant download (7 files, Llama-3.2-1B)             | 10-20 min  |
| 5    | TRT-LLM engine build (fp16 + int8 + int4)               | 45-90 min  |

Expected layout on success:
```
${PROFILE_ROOT}/models/hf_full/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<hash>/
${PROFILE_ROOT}/models/hf_full/models--Qwen--Qwen3-4B/snapshots/<hash>/
${PROFILE_ROOT}/models/gguf/Llama-3.2-1B-Instruct-{F16,Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_0,Q3_K_L}.gguf
${PROFILE_ROOT}/models/trtllm_engines/llama-3.2-1b-instruct{,-int8,-int4}/rank0.engine
```

If any step fails, re-invoke `prepare_orin.sh` — completed steps are
skipped by cache checks.

### 3. Run the full collection — `run_orin_collection.sh`

```bash
bash scripts/collect/orin/run_orin_collection.sh
```

Runs, in order (`stage1 | side | stage2 | stage3 | build`) — this one
command produces **every Orin-side file the plotting scripts read**:

1. **Stage 1 — master sweep** (~7 h)
   `sweep/sweep.sh` (run with `SWEEP_SCOPE=full`) iterates
   `pp ∈ {128, 256, 512, 1024, 2048, 4096} × gen ∈ …` for each
   `(framework, quantization)` across all 5 frameworks (incl. SGLang) and
   the Llama-3.2-1B / Llama-3.1-8B / Mixtral models. One measured run per
   cell (with 3 warm-ups); the row lands in
   `sweep_results/sweep_locked_master.csv` as it completes. Interrupting
   is safe — resume by re-invoking the same command; existing rows are
   skipped.

2. **Side — extra paper-series grids** (~1 h + ~8-10 h longctx)
   `side_sweeps.sh` runs the three small grids behind the remaining paper
   series: torch.compile (8 cells), llama.cpp flash-attn ON (16 cells),
   and the long-decode extension out to 131 K generated tokens (27 cells —
   the slow part; single 131 K decodes take ~1 h each). The longctx TRT
   cells build a long-context engine on demand (cached, reuses prepare's
   fp16 checkpoint). Subset with `SIDE=compile|fa|longctx`.

3. **Stage 2 — repeatability** (~7 h × `$REP_COUNT`, default 15)
   The **same worker** re-run with `SWEEP_SCOPE=1b` — identical cells to
   stage 1 except the Llama-3.1-8B and Mixtral blocks are skipped (only
   the Llama-3.2-1B cells are repeated). One fresh container-loaded pass
   per rep, one CSV per rep, to populate the cross-execution CV story
   (AE §6 "N=15 means match a single-run sweep within 0.6%"). Common
   overrides:
   ```bash
   REP_COUNT=3  bash scripts/collect/orin/run_orin_collection.sh stage2   # ~21 h anchor set
   REP_COUNT=1  bash scripts/collect/orin/run_orin_collection.sh stage2   # ~7 h smoke test
   ```

4. **Stage 3 — nsys kernel decomposition** (~2 h)
   Captures anchor cells under Nsight Systems, extracts per-op /
   per-category / launch-gap JSONs into `data/nsys/` (backs Fig 4 / 9 / 10
   on Orin). Requires `nsys` on the host (part of the JetPack CUDA
   install). Thor's Stage 3 is graphs-ON re-run — Orin's Stage 3 is nsys
   instead because sm_87 has no stable graphs path (METHODOLOGY_NOTES.md).

5. **Build** (~5 min)
   `build_sweep_locked_orin.py` concatenates the master sweep and all rep
   CSVs into a single raw `data/chat/sweep_locked.csv` (~ `REP_COUNT + 1`
   rows / cell; plot loaders average per cell) and regenerates
   `data/chat/mbu_pp512_gen256.csv` (Orin rows; Thor rows preserved) plus
   the three side files from the side stage:
   `data/chat/pytorch_compile.csv`, `data/chat/llamacpp_fa_orin.csv`,
   `data/chat/longctx_fp16_orin.csv`. Side files whose raw sweep is absent
   are left at their shipped contents.

Skip individual stages:
```bash
NSYS=0 bash run_orin_collection.sh                      # everything but nsys
bash run_orin_collection.sh stage3                      # just nsys
SIDE=compile bash run_orin_collection.sh side           # just the compile grid
bash run_orin_collection.sh build                       # just the fold-in
```

**Agentic (SWE-bench-live, §VI / Fig 12) is NOT part of the main pipeline** —
same shape as Thor. Run the per-framework drivers separately:
```bash
bash scripts/collect/orin/agentic/rerun_swebench_live_{vllm,sglang,trtllm,llamacpp,pytorch}.sh
```

### 4. Regenerate every figure

```bash
mkdir -p figs
for s in scripts/plot/gen_fig*.py; do
    python3 "$s" --out figs/
done
```

Each script reads only `data/chat/*.csv` and `data/nsys/*.json` — no
network, no GPU. Output PNGs land in `figs/`.

### 5. Verify against the shipped CSVs (optional but recommended)

`verify.py` performs 17 spot-checks against the shipped baseline. Point it
at the newly built `sweep_locked.csv` — every check passes if the newly
collected data is within paper tolerance:

```bash
python3 verify.py --sweep data/chat/sweep_locked.csv
```

Typical delta vs the shipped CSV (paper's cross-execution CV distribution):

| metric        | mean CV | p90 CV | max CV |
|---------------|---------|--------|--------|
| TPOT          | 0.79%   | 2.15%  | 6.04%  |
| TTFT          | 0.69%   | 1.86%  | 6.76%  |
| decode power  | 0.47%   | 1.06%  | 3.57%  |
| decode energy | 0.51%   | 1.22%  | 4.13%  |

Values outside this envelope usually mean (a) clocks are not locked
(re-run `preflight.sh`) or (b) something else was using the GPU during
the sweep.

### 6. Troubleshooting

| Symptom                              | Cause / fix                                                                             |
|--------------------------------------|-----------------------------------------------------------------------------------------|
| `ERROR: swap in use (X KB > 5 GB)`   | `sudo swapoff -a && sudo swapon -a`, then re-run preflight.                             |
| `basename: missing operand`          | HF snapshot dir is empty — re-run `prepare_orin.sh` step 3.                             |
| `run_vllm: command not found`        | You edited `sweep.sh` and broke a function definition. Reset via `git checkout`.        |
| `NvMap OOM` on vLLM GGUF long ctx    | Known Orin sm_87 limitation — that cell is skipped in the shipped data (see             |
|                                      | `data/rebuttal/q1_repeatability/missing_cells.csv`).                                    |
| `NVML_SUCCESS INTERNAL ASSERT`       | vLLM V0 + graphs is infeasible on Orin. Do not set `VLLM_ENFORCE_EAGER=0` for V0.       |
| Decode throughput 25% lower than paper for llama.cpp | Ensure `-fa` (flash-attn) is on. The default sweep already sets this.    |

---

## Thor (AGX Thor 128 GB)

The Thor pipeline mirrors Orin. Full details are in `scripts/collect/thor/README.md`.
Short form:

```bash
export PROFILE_ROOT=/path/to/thor/profile          # holds data/benchmarks + data/models + work/
bash scripts/collect/thor/run_thor_collection.sh    # stages 1-3 + build
```

Runs stages: master sweep (eager, num_runs=1) → N=15 repeatability →
throughput graphs-ON (vLLM/SGLang GGUF only) → fold into
`data/chat/sweep_locked_thor.csv`. Per-platform graph-capture policy is
documented in `METHODOLOGY_NOTES.md` — Thor turns graphs ON for the GGUF
throughput cells to match published paper numbers.

---

## Frequently asked

### Why does Workflow A produce figures that differ from the paper?

Workflow A only re-renders the plots from the CSVs already in `data/`. It
should not produce different figures unless the plot scripts were edited.
If you see structural differences, check:

1. You are reading `data/chat/sweep_locked.csv` (Orin) or
   `sweep_locked_thor.csv` (Thor), not an older cached CSV.
2. `llamacpp_fa` (flash-attn variant) is filtered in — this is the
   variant the paper uses.  See `scripts/plot/_fits_common.py:CANON`.
3. For Fig 11 the Thor points read `data/chat/pareto_thor/pareto_thor_base.csv`
   (graphs-ON for the GGUF cells), not `sweep_locked_thor.csv`.

If figures still differ numerically, re-run Workflow B on that platform to
regenerate the CSV — hardware or software drift can shift numbers by a
few percent even under the same clock lock.

### Which badges does each workflow support?

| workflow                     | Available | Functional | Reproduced |
|------------------------------|-----------|------------|------------|
| A (plot-only on shipped CSV) | ✓         | ✓          |            |
| B (full re-collect + plot)   | ✓         | ✓          | ✓          |

### What if I only have Orin (or only Thor)?

Each platform is self-contained. Only the Fig 11 pareto and the Fig 3 MBU
bar show side-by-side Orin+Thor points; every other figure is
per-platform. Missing-platform points are dropped, not fabricated.
