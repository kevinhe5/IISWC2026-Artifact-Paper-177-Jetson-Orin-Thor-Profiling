# Methodology notes — Orin+Thor artifact

Cross-referenced from the paper (§methodology / figure captions) and from
individual data-dir READMEs.

## CUDA-graph capture status — throughput figures (Fig 11, Tab 4, Fig 2)

Per-framework, per-platform graph-capture status for the throughput figures:

| Framework    | Thor (sm_110)                                       | Orin (sm_87)                                                                                    |
|--------------|-----------------------------------------------------|--------------------------------------------------------------------------------------------------|
| vLLM         | graphs-ON for gguf_Q4_0, gguf_Q4_K_M, gguf_Q8_0; eager for fp16 | eager (GGUF path silently falls back to V0 where graph capture is infeasible) |
| SGLang       | graphs-ON for gguf points with FlashInfer           | eager (no stable FlashInfer path on sm_87)                                                       |
| TensorRT-LLM | AOT-compiled (always graph-captured)                | AOT-compiled (always graph-captured)                                                             |
| llama.cpp    | native C++ scheduler (no CUDA graphs)               | native C++ scheduler (no CUDA graphs)                                                            |
| PyTorch      | eager (or torch.compile for pytorch_compile cells)  | eager (or torch.compile for pytorch_compile cells; see `data/chat/pytorch_compile.csv`)          |


**Latency and energy fits (Fig 5, 6, 8):** run eager on both platforms
with N=15 independent runs per cell for cross-platform consistency.

## Power measurement — Orin

Sampled at the tegrastats effective minimum (~10 Hz, tegrastats clamps
below ~100 ms) from Orin's two on-board INA3221 monitors:

- INA3221 #1 (via tegrastats): VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0 (5V
  peripheral rail, **not** the board input)
- INA3221 #2 (via hwmon sysfs): VDDQ_VDD2_1V8AO (LPDDR5 DRAM cell)

Reported "total" power = 4-rail sum. The 19V board input at the DC
barrel jack is not instrumented; PMIC switching losses and board-level
regulation overhead (~5-7 W during decode) are NOT captured. External
wall-meter readings run 10-15% higher than the 4-rail sum.

## Framework version pinning

See `manifests/framework_versions.csv` for the exact library
versions and build knobs per platform.

Notable cross-platform version deltas:
- vLLM: Thor 0.12.0 / Orin 0.8.6
- SGLang: Thor 0.5.7 / Orin 0.4.6.post2
- transformers (PyTorch eager): Thor 5.11.0 (torch 2.10.0) / Orin 4.57.3
  (torch 2.6.0)
- bitsandbytes: Thor 0.49.0 / Orin 0.45.4.dev0
- llama-cpp-python: Thor 0.3.16 / Orin 0.3.8 (different bindings, both
  wrapping a llama.cpp b52** C++ core)

## See also

- `data/contention/summary_thor.csv` — contention slowdown data
  backing Table 5.
