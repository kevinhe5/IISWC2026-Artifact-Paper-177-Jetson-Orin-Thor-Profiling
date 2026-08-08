#!/usr/bin/env python3
"""Per-framework EdgeReasoning-style analytical fits for §IV.A / §IV.B.

For each framework f at a chosen quantization (fp16 by default), fits four
closed-form scaling models from the chat-sweep CSV and emits
publication-ready PDFs + a coefficient table:

  Prefill latency      L_pref(I) = a I_pad^2 + b I_pad + c     (Fig 2 form)
  Decode latency       L_dec(I, O) = n O + m (I O + O(O-1)/2)  (Eq 2 form)
                       => TPOT(I, O) = m * (I + (O-1)/2) + n   (per-tok form)
  Decode power         P_dec(O) = y ln(O) + z   for O >= 64    (Fig 5 form)
  Decode energy / tok  E_tok(O) = P_dec(O) * TPOT              (Fig 5 form)

I_pad = ceil(I / 128) * 128 (Tensor-Core padding). The MAPE on held-out
(pp, gen) cells is computed and printed alongside each fit.

Outputs (relative to the JetsonAnalysis root):
  figs/fig_v5_ttft_fits.{pdf,png}      — TTFT vs I, one curve per framework
  figs/fig_v5_tpot_fits.{pdf,png}      — TPOT vs effective context I + O/2
  figs/fig_v5_power_fits.{pdf,png}     — P_dec vs gen, log-linear in O
  figs/fig_v5_energy_fits.{pdf,png}    — E_tok vs gen
  figs/scripts/_perf_fit_table.tex     — coefficient table for §IV.A/§IV.B
  figs/scripts/_perf_fit_residuals.txt — fitted coefficients (for log)

Run from JetsonAnalysis/:
    python3 figs/scripts/fit_perf_models.py [--quant 16-bit] [--model Llama-3.2-1B]
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
# Source CSVs (kept for traceability — data below was extracted from these):
#   data/sweep_results/sweep_locked_20260428_020532.csv          (5 fw)
#   jetson-containers/.../pytorch_compile_sweep_*.csv            (compile variant)
DEFAULT_CSV = _HERE.parents[1] / "data" / "sweep_results" / "sweep_locked_20260428_020532.csv"

# =====================================================================
# Self-contained data snapshot.  Llama-3.2-1B, 16-bit (fp16/bf16), AGX
# Orin 32 GB locked clocks.  All four downstream fits (TTFT, TPOT, power,
# energy) consume only these tuples — no runtime CSV dependency.
# Each tuple = (I, O, ttft_ms, tpot_ms, dec_w, e_tok_mj).
# None entries mean the field was missing in the source CSV.
# =====================================================================
DATA_16BIT = {
    "trtllm": [
        (   128,    128,    24.63,   16.28,  35.03,   656.32),
        (   128,    256,    23.12,   16.30,  37.72,   708.57),
        (   128,    512,    23.06,   16.22,  38.70,   725.35),
        (   128,   1024,    23.11,   16.28,  39.61,   745.28),
        (   128,   2048,    23.12,   16.38,  40.03,   757.77),
        (   128,   4096,    23.73,   16.59,  40.19,   771.27),
        (   128,  16384,    23.48,   17.73,  47.14,   835.94),
        (   128,  32768,    23.06,   19.27,  47.53,   916.11),
        (   128,  65536,    22.91,   22.32,  48.08,  1073.32),
        (   128, 131072,    22.74,   28.43,  48.77,  1386.26),
        (   256,    128,    28.99,   16.20,  35.68,   662.96),
        (   256,    256,    28.11,   16.18,  38.12,   710.19),
        (   256,    512,    28.31,   16.28,  39.18,   735.88),
        (   256,   1024,    28.18,   16.30,  39.77,   748.66),
        (   256,   2048,    28.81,   16.44,  39.96,   759.53),
        (   256,   4096,    28.38,   16.62,  40.22,   772.74),
        (   512,    128,    38.84,   16.27,  35.86,   668.61),
        (   512,    256,    38.64,   16.25,  37.94,   710.46),
        (   512,    512,    38.93,   16.23,  39.30,   735.94),
        (   512,   1024,    39.44,   16.30,  39.84,   750.25),
        (   512,   2048,    39.13,   16.45,  40.07,   761.72),
        (   512,   4096,    38.73,   16.61,  40.23,   773.19),
        (  1024,    128,    72.78,   16.32,  36.95,   689.75),
        (  1024,    256,    73.13,   16.30,  38.62,   723.88),
        (  1024,    512,    72.52,   16.31,  39.44,   741.77),
        (  1024,   1024,    73.08,   16.38,  39.88,   754.33),
        (  1024,   2048,    72.62,   16.47,  40.12,   763.72),
        (  1024,   4096,    72.62,   16.67,  40.25,   775.99),
        (  2048,    128,   102.89,   16.48,  37.59,   707.19),
        (  2048,    256,   102.64,   16.41,  38.90,   733.67),
        (  2048,    512,   102.89,   16.43,  39.62,   750.25),
        (  2048,   1024,   102.90,   16.48,  40.04,   761.64),
        (  2048,   2048,   103.44,   16.59,  40.20,   770.63),
        (  2048,   4096,   103.05,   16.76,  40.30,   781.63),
        (  4096,    128,   215.22,   16.68,  39.16,   741.22),
        (  4096,    256,   215.06,   16.62,  39.69,   755.95),
        (  4096,    512,   214.76,   16.63,  40.01,   765.66),
        (  4096,   1024,   212.97,   16.67,  40.18,   772.93),
        (  4096,   2048,   212.99,   16.79,  40.30,   781.58),
        (  4096,   4096,   214.37,   16.96,  40.35,   791.74),
    ],
    "vllm": [
        (   128,    128,    27.19,   20.57,  37.64,   768.24),
        (   128,    128,    34.41,   19.34,  40.22,   771.96),
        (   128,    256,    24.41,   20.02,  41.69,   831.41),
        (   128,    256,    35.07,   20.13,  41.23,   826.79),
        (   128,    512,    24.24,   19.70,  42.96,   844.57),
        (   128,    512,    35.01,   20.45,  41.96,   856.58),
        (   128,   1024,    27.14,   20.61,  42.15,   867.64),
        (   128,   1024,    34.45,   20.51,  42.46,   870.16),
        (   128,   2048,    24.80,   20.39,  42.96,   875.57),
        (   128,   2048,    35.23,   20.54,  42.83,   879.06),
        (   128,   4096,    25.31,   20.61,  43.33,   892.92),
        (   128,   4096,    34.75,   20.52,  43.33,   888.60),
        (   128,   8192,    35.39,   20.06,  44.98,   902.15),
        (   128,  16384,    34.55,   20.14,  46.50,   936.53),
        (   128,  32768,    35.03,   21.60,  48.14,  1039.67),
        (   128,  65536,    34.71,   24.30,  50.10,  1217.56),
        (   128, 130944,    34.79,   30.48,  51.87,  1580.97),
        (   256,    128,    25.06,   20.53,  38.60,   786.38),
        (   256,    128,    35.68,   19.54,  40.23,   779.82),
        (   256,    256,    27.00,   21.25,  40.55,   858.33),
        (   256,    256,    35.58,   19.79,  41.84,   824.88),
        (   256,    512,    25.55,   20.27,  41.98,   848.98),
        (   256,    512,    35.44,   19.96,  42.59,   848.57),
        (   256,   1024,    25.15,   20.41,  42.35,   863.71),
        (   256,   1024,    35.26,   20.43,  42.64,   870.25),
        (   256,   2048,    25.03,   20.43,  42.52,   868.25),
        (   256,   2048,    35.64,   20.61,  43.34,   892.76),
        (   256,   4096,    25.42,   20.79,  43.00,   893.69),
        (   256,   4096,    35.43,   20.67,  43.20,   892.90),
        (   512,    128,    25.78,   20.66,  38.42,   787.70),
        (   512,    128,    50.98,   20.54,  39.81,   811.14),
        (   512,    256,    26.03,   20.83,  40.12,   832.28),
        (   512,    256,    50.49,   20.42,  41.31,   840.41),
        (   512,    512,    25.42,   20.57,  41.70,   856.01),
        (   512,    512,    50.49,   20.60,  41.95,   862.46),
        (   512,   1024,    27.13,   20.05,  42.98,   860.86),
        (   512,   1024,    50.71,   19.78,  43.57,   861.10),
        (   512,   2048,    26.54,   20.80,  42.59,   885.43),
        (   512,   2048,    50.49,   20.27,  43.35,   878.38),
        (   512,   4096,    24.84,   19.93,  43.90,   874.75),
        (   512,   4096,    50.26,   20.48,  43.51,   890.63),
        (  1024,    128,    26.30,   19.90,  39.09,   771.92),
        (  1024,    128,    73.43,   20.13,  40.77,   814.21),
        (  1024,    256,    26.02,   19.94,  41.49,   824.29),
        (  1024,    256,    73.68,   20.59,  41.55,   852.10),
        (  1024,    512,    27.34,   20.74,  41.27,   854.32),
        (  1024,    512,    72.74,   20.44,  42.42,   865.44),
        (  1024,   1024,    26.19,   19.91,  43.28,   860.86),
        (  1024,   1024,    72.48,   19.93,  43.58,   867.61),
        (  1024,   2048,    26.02,   20.19,  43.45,   877.05),
        (  1024,   2048,    72.95,   19.90,  43.97,   874.69),
        (  1024,   4096,    26.54,   20.55,  43.14,   886.50),
        (  1024,   4096,    73.29,   19.95,  44.36,   885.01),
        (  2048,    128,    27.74,   20.45,  38.99,   791.10),
        (  2048,    128,   114.54,   20.02,  42.26,   839.41),
        (  2048,    256,    28.59,   20.84,  40.73,   845.43),
        (  2048,    256,   115.08,   19.46,  43.65,   846.27),
        (  2048,    512,    28.40,   20.47,  41.87,   855.26),
        (  2048,    512,   114.24,   20.67,  42.50,   876.59),
        (  2048,   1024,    28.19,   20.50,  42.35,   867.31),
        (  2048,   1024,   114.25,   20.74,  42.81,   887.17),
        (  2048,   2048,    27.53,   19.91,  43.73,   870.29),
        (  2048,   2048,   114.11,   19.92,  44.25,   880.94),
        (  2048,   4096,    27.95,   20.78,  43.40,   901.40),
        (  2048,   4096,   114.27,   20.69,  43.57,   901.32),
        (  4096,    128,    31.15,   20.23,  39.49,   792.47),
        (  4096,    128,   237.64,   20.47,  43.27,   878.93),
        (  4096,    256,    31.76,   20.23,  41.41,   834.55),
        (  4096,    256,   234.54,   21.02,  42.73,   894.78),
        (  4096,    512,    31.38,   20.34,  42.52,   863.07),
        (  4096,    512,   234.73,   20.54,  43.53,   892.44),
        (  4096,   1024,    31.43,   20.63,  42.61,   878.24),
        (  4096,   1024,   237.92,   20.56,  43.62,   895.95),
        (  4096,   2048,    30.36,   20.00,  43.95,   878.59),
        (  4096,   2048,   234.71,   19.65,  45.13,   886.63),
        (  4096,   4096,    31.35,   20.68,  43.98,   909.13),
        (  4096,   4096,   237.79,   20.55,  44.24,   908.86),
    ],
    "sglang": [
        (   128,    128,    47.86,   22.75,  37.69,   850.69),
        (   128,    128,    51.11,   23.21,  37.47,   862.98),
        (   128,    256,    48.22,   22.98,  38.71,   886.03),
        (   128,    256,    50.60,   23.72,  38.59,   911.56),
        (   128,    512,    47.96,   23.30,  39.38,   915.89),
        (   128,    512,    51.65,   23.18,  39.92,   923.43),
        (   128,   1024,    49.38,   23.68,  39.54,   935.40),
        (   128,   1024,    50.85,   23.45,  40.06,   938.70),
        (   128,   2048,    48.76,   22.81,  40.52,   923.60),
        (   128,   2048,    48.96,   23.36,  40.18,   938.12),
        (   128,   4096,    46.64,   22.78,  40.95,   932.43),
        (   128,   4096,    49.14,   22.99,  40.64,   933.98),
        (   128,   8192,    48.35,   23.28,  41.24,   959.88),
        (   128,  16384,    46.48,   23.02,  42.59,   980.64),
        (   128,  32768,    47.35,   22.69,  46.28,  1050.16),
        (   128,  65536,    47.37,   24.94,  48.48,  1209.05),
        (   128, 130560,    47.15,   31.59,  49.44,  1561.56),
        (   256,    128,    47.41,   22.77,  37.52,   847.58),
        (   256,    128,    49.98,   22.79,  37.85,   855.99),
        (   256,    256,    47.37,   23.26,  38.64,   895.11),
        (   256,    256,    47.80,   22.91,  38.80,   885.46),
        (   256,    512,    48.38,   23.10,  39.34,   906.98),
        (   256,    512,    51.38,   23.90,  39.31,   937.74),
        (   256,   1024,    49.52,   23.37,  39.48,   921.83),
        (   256,   1024,    54.32,   23.20,  39.78,   922.15),
        (   256,   2048,    48.31,   23.27,  40.00,   930.51),
        (   256,   2048,    48.64,   22.79,  40.52,   922.82),
        (   256,   4096,    44.80,   22.91,  40.72,   932.90),
        (   256,   4096,    49.50,   23.52,  40.04,   941.68),
        (   512,    128,    48.12,   23.63,  37.62,   882.11),
        (   512,    128,    48.79,   23.10,  37.11,   850.69),
        (   512,    256,    47.59,   23.32,  39.00,   905.97),
        (   512,    256,    48.75,   22.99,  38.71,   886.34),
        (   512,    512,    46.49,   23.32,  39.69,   924.03),
        (   512,    512,    49.58,   23.08,  39.47,   909.26),
        (   512,   1024,    47.39,   22.72,  40.70,   924.06),
        (   512,   1024,    47.98,   23.97,  39.63,   948.83),
        (   512,   2048,    46.45,   23.43,  40.27,   943.14),
        (   512,   2048,    48.76,   23.27,  40.03,   931.01),
        (   512,   4096,    49.31,   23.12,  41.09,   949.76),
        (   512,   4096,    52.52,   23.56,  40.85,   962.26),
        (  1024,    128,    51.68,   23.64,  37.27,   874.10),
        (  1024,    128,    71.99,   23.78,  40.14,   946.91),
        (  1024,    256,    53.16,   23.04,  38.68,   887.80),
        (  1024,    256,    71.05,   22.69,  39.76,   898.82),
        (  1024,    512,    50.27,   23.25,  39.21,   910.11),
        (  1024,    512,    70.32,   22.93,  39.96,   914.54),
        (  1024,   1024,    48.65,   23.08,  39.87,   919.37),
        (  1024,   1024,    70.40,   23.00,  40.15,   922.63),
        (  1024,   2048,    48.50,   22.99,  40.43,   928.90),
        (  1024,   2048,    71.37,   22.78,  40.68,   926.28),
        (  1024,   4096,    49.56,   23.20,  40.51,   939.60),
        (  1024,   4096,    70.73,   23.20,  40.59,   941.53),
        (  2048,    128,    51.04,   23.06,  37.46,   856.99),
        (  2048,    128,   115.52,   22.97,  39.65,   903.75),
        (  2048,    256,    50.36,   23.23,  38.80,   897.63),
        (  2048,    256,   115.06,   23.02,  39.98,   916.82),
        (  2048,    512,    50.22,   23.07,  39.59,   911.40),
        (  2048,    512,   115.59,   23.32,  39.90,   928.80),
        (  2048,   1024,    49.92,   22.91,  40.34,   923.20),
        (  2048,   1024,   114.60,   23.28,  40.13,   933.24),
        (  2048,   2048,    52.52,   23.26,  40.30,   936.77),
        (  2048,   2048,   115.29,   23.29,  40.29,   937.89),
        (  2048,   4096,    51.83,   23.43,  40.47,   947.78),
        (  2048,   4096,   115.34,   22.93,  41.02,   940.37),
        (  4096,    128,    54.10,   23.17,  37.88,   870.69),
        (  4096,    128,   224.43,   23.08,  41.57,   952.03),
        (  4096,    256,    54.62,   23.48,  38.83,   908.25),
        (  4096,    256,   224.61,   23.32,  40.77,   947.25),
        (  4096,    512,    54.51,   23.61,  39.40,   928.29),
        (  4096,    512,   223.61,   23.37,  40.63,   947.68),
        (  4096,   1024,    53.81,   22.87,  40.64,   928.56),
        (  4096,   1024,   224.32,   22.93,  40.96,   938.14),
        (  4096,   2048,    54.24,   23.10,  40.78,   941.43),
        (  4096,   2048,   225.28,   23.32,  40.78,   950.42),
        (  4096,   4096,    53.83,   23.22,  41.20,   956.35),
        (  4096,   4096,   226.21,   23.57,  41.51,   978.05),
    ],
    "llamacpp": [
        (   128,    128,    33.17,   21.24,  35.88,   851.01),
        (   128,    256,    33.29,   21.30,  38.41,   915.32),
        (   128,    512,    33.17,   21.30,  39.67,   946.36),
        (   128,   1024,    33.28,   21.35,  40.30,   964.58),
        (   128,   2048,    33.17,   21.43,  40.87,   981.87),
        (   128,   4096,    33.22,   21.94,  41.26,  1013.67),
        (   128,   8192,    32.96,   23.11,  46.44,  1073.12),
        (   128,  16384,    32.88,   25.46,  46.98,  1196.00),
        (   128,  32768,    33.25,   30.64,  48.50,  1486.14),
        (   128,  65536,    33.90,   44.20,  52.24,  2309.04),
        (   128, 131072,    34.11,   71.67,  54.25,  3888.23),
        (   256,    128,    54.75,   21.76,  36.30,   879.77),
        (   256,    256,    54.55,   21.71,  38.59,   935.53),
        (   256,    512,    54.85,   21.37,  39.78,   951.65),
        (   256,   1024,    55.11,   21.28,  40.50,   965.76),
        (   256,   2048,    54.94,   21.48,  40.92,   985.22),
        (   256,   4096,    54.67,   21.97,  41.33,  1016.58),
        (   512,    128,   107.27,   21.09,  37.11,   872.28),
        (   512,    256,   107.30,   21.00,  39.17,   919.70),
        (   512,    512,   107.43,   21.15,  40.17,   951.45),
        (   512,   1024,   107.47,   21.31,  40.67,   971.24),
        (   512,   2048,   107.61,   21.53,  41.02,   990.03),
        (   512,   4096,   107.76,   22.05,  41.38,  1021.78),
        (  1024,    128,   250.17,   21.41,  37.74,   898.89),
        (  1024,    256,   250.12,   21.40,  39.52,   944.52),
        (  1024,    512,   250.32,   21.48,  40.28,   968.91),
        (  1024,   1024,   248.93,   21.58,  40.73,   984.88),
        (  1024,   2048,   250.16,   21.81,  41.15,  1005.69),
        (  1024,   4096,   248.81,   22.29,  41.52,  1035.62),
        (  2048,    128,   595.86,   21.90,  38.61,   939.58),
        (  2048,    256,   594.10,   21.97,  39.90,   978.12),
        (  2048,    512,   595.63,   21.94,  40.69,   998.57),
        (  2048,   1024,   594.77,   22.01,  41.17,  1014.17),
        (  2048,   2048,   595.73,   22.30,  41.45,  1033.80),
        (  2048,   4096,   595.70,   22.83,  41.66,  1063.21),
        (  4096,    128,  1579.21,   22.92,  39.12,   994.71),
        (  4096,    256,  1584.39,   22.76,  40.52,  1027.31),
        (  4096,    512,  1582.82,   22.78,  41.22,  1047.66),
        (  4096,   1024,  1584.58,   22.97,  41.55,  1065.27),
        (  4096,   2048,  1583.03,   23.33,  41.70,  1086.64),
        (  4096,   4096,  1584.31,   23.93,  41.89,  1118.94),
    ],
    # llama.cpp fp16 with flash-attn ON (`-fa`). Re-measured 2026-05-21 from
    # run_fa_sweep_full_f16.sh -> fa_prefill_sweep_f16_20260521_171639.csv
    # (prefill, gen=32, I=128..16384) + fa_longdecode_f16_20260521_171639.csv
    # (decode, I=128, O=4096/16384/32768). Power/energy not measured here, so
    # dec_w / e_tok are None and this series only appears on the TTFT/TPOT
    # plots. The FA-off counterpart is the "llamacpp" series above.
    "llamacpp_faon": [
        (   128,     32,     30.64,  20.831, None, None),
        (   128,   4096,     30.64,  20.499, None, None),
        (   128,  16384,     30.61,  22.097, None, None),
        (   128,  32768,     30.72,  24.409, None, None),
        (   256,     32,     45.89,  21.041, None, None),
        (   512,     32,     78.74,  20.813, None, None),
        (   768,     32,    118.98,  21.026, None, None),
        (  1024,     32,    151.78,  21.240, None, None),
        (  1536,     32,    226.14,  21.264, None, None),
        (  2048,     32,    303.45,  21.412, None, None),
        (  3072,     32,    464.31,  21.723, None, None),
        (  4096,     32,    638.01,  21.987, None, None),
        (  6144,     32,   1004.49,  22.686, None, None),
        (  8192,     32,   1411.61,  23.697, None, None),
        ( 12288,     32,   2339.26,  24.049, None, None),
        ( 16384,     32,   3406.59,  25.157, None, None),
    ],
    "pytorch": [
        (   128,    128,    51.53,   39.85,  28.56,  1128.97),
        (   128,    256,    47.05,   36.63,  30.29,  1105.21),
        (   128,    512,    47.29,   37.88,  30.23,  1142.77),
        (   128,   1024,    47.29,   37.48,  30.80,  1152.98),
        (   128,   2048,    47.13,   37.87,  31.25,  1182.79),
        (   128,   4096,    47.84,   38.06,  32.78,  1247.33),
        (   128,   8192,    49.99,   38.63,  33.77,  1304.39),
        (   128,  16384,    45.76,   36.47,  38.03,  1387.12),
        (   128,  32768,    46.10,   41.35,  41.28,  1706.71),
        (   128,  65536,    45.51,   56.78,  42.78,  2428.71),
        (   256,    128,    49.73,   37.29,  29.75,  1100.79),
        (   256,    256,    49.84,   37.34,  30.30,  1127.03),
        (   256,    512,    49.70,   36.86,  30.98,  1139.75),
        (   256,   1024,    49.18,   37.26,  31.69,  1179.70),
        (   256,   2048,    50.40,   36.60,  32.01,  1171.18),
        (   256,   4096,    50.18,   37.33,  32.52,  1213.48),
        (   512,    128,    72.67,   36.60,  30.23,  1097.74),
        (   512,    256,    73.29,   37.93,  30.41,  1148.77),
        (   512,    512,    73.06,   36.84,  31.30,  1150.68),
        (   512,   1024,    73.31,   36.64,  31.72,  1160.82),
        (   512,   2048,    73.47,   36.94,  32.02,  1182.20),
        (   512,   4096,    73.30,   37.07,  32.82,  1216.24),
        (  1024,    128,   143.70,   37.10,  30.70,  1129.93),
        (  1024,    256,   140.78,   37.63,  30.90,  1158.17),
        (  1024,    512,   140.72,   36.57,  31.89,  1163.91),
        (  1024,   1024,   140.37,   37.27,  31.84,  1185.40),
        (  1024,   2048,   154.38,   37.33,  32.30,  1205.26),
        (  1024,   4096,   142.03,   37.92,  32.83,  1244.50),
        (  2048,    128,   279.36,   37.40,  31.72,  1176.88),
        (  2048,    256,   279.90,   37.07,  32.18,  1188.20),
        (  2048,    512,   279.26,   36.74,  32.53,  1192.82),
        (  2048,   1024,   280.39,   37.88,  32.34,  1224.00),
        (  2048,   2048,   278.33,   36.66,  33.45,  1225.68),
        (  2048,   4096,   278.88,   37.58,  33.91,  1274.17),
        (  4096,    128,   575.38,   37.56,  33.60,  1252.31),
        (  4096,    256,   576.11,   37.60,  33.64,  1260.05),
        (  4096,    512,   575.78,   37.46,  34.15,  1276.67),
        (  4096,   1024,   574.13,   37.65,  34.24,  1287.83),
        (  4096,   2048,   574.39,   37.01,  34.97,  1293.31),
        (  4096,   4096,   577.66,   37.29,  35.73,  1332.15),
    ],
    "pytorch_compile": [
        (   128,    128,    23.71,   17.13,  33.79,   578.86),
        (   128,    256,    24.08,   16.90,  36.53,   617.37),
        (   128,    512,    23.32,   17.27,  37.61,   649.53),
        (   128,   1024,    23.56,   18.08,  39.75,   718.72),
        (   128,   2048,    23.17,   19.57,  41.38,   809.68),
        (   128,   4096,    22.99,   22.58,  43.81,   989.27),
        (   128,   8192,    23.96,   28.28,  46.52,  1315.88),
        (   128,  16384,    23.95,   43.15,  48.72,  2101.90),  # measured 20260617, clean (no stress); 32768+ OOM (default attn, no flash-attn)
        (   256,    128,    35.77,   17.02,  34.04,   579.40),
        (   512,    128,    63.69,   17.85,  34.49,   615.61),
        (   512,    256,    64.75,   18.09,  36.66,   663.22),
        (  1024,    128,   138.12,   19.28,  35.35,   681.35),
        (  2048,    128,   323.67,   22.19,  36.67,   813.79),
        (  4096,    128,   878.79,   28.09,  38.57,  1083.37),
    ],
}
OUT_DIR = _HERE.parent  # JetsonAnalysis/figs/

# Mirrors _viz_app.QUANT_ALIASES.
QUANT_ALIASES = {
    "bf16": "16-bit", "f16": "16-bit", "fp16": "16-bit",
    "fp16_mb32": "16-bit", "fp16_nocache": "16-bit",
    "4bit": "int4", "int4": "int4",
    "8bit": "int8", "int8": "int8",
    "Q8_0": "Q8_0", "q8_0": "Q8_0", "gguf_Q8_0": "Q8_0",
    "Q4_K_M": "Q4_K_M", "q4_k_m": "Q4_K_M",
    "gguf_Q4_K_M": "Q4_K_M", "Q4_K_M_gpu16": "Q4_K_M",
    "Q4_0": "Q4_0", "q4_0": "Q4_0", "gguf_Q4_0": "Q4_0",
    "Q6_K": "Q6_K", "q6_k": "Q6_K", "gguf_Q6_K": "Q6_K",
    "Q5_K_M": "Q5_K_M", "q5_k_m": "Q5_K_M", "gguf_Q5_K_M": "Q5_K_M",
    "Q3_K_M": "Q3_K_M", "q3_k_m": "Q3_K_M", "gguf_Q3_K_M": "Q3_K_M",
    "Q3_K_L": "Q3_K_L", "q3_k_l": "Q3_K_L", "gguf_Q3_K_L": "Q3_K_L",
}

FW_ORDER = ["trtllm", "vllm", "sglang", "llamacpp", "llamacpp_faon",
            "pytorch", "pytorch_compile"]
FW_LABEL = {
    "trtllm":         "TensorRT-LLM",
    "vllm":           "vLLM",
    "sglang":         "SGLang",
    "llamacpp":       "llama.cpp (FA off)",
    "llamacpp_faon":  "llama.cpp (FA on)",
    "pytorch":        "PyTorch (eager)",
    "pytorch_compile":"PyTorch + torch.compile",
}
FW_COLOR = {
    "trtllm":         "#1f77b4",
    "vllm":           "#ff7f0e",
    "sglang":         "#9467bd",
    "llamacpp":       "#e377c2",
    "llamacpp_faon":  "#be185d",
    "pytorch":        "#2ca02c",
    "pytorch_compile":"#15803d",
}
FW_MARKER = {
    "trtllm":         "o",
    "vllm":           "s",
    "sglang":         "^",
    "llamacpp":       "D",
    "llamacpp_faon":  "*",
    "pytorch":        "v",
    "pytorch_compile":"x",
}


def fnum(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def norm_quant(q):
    return QUANT_ALIASES.get(q, q) if q else "?"


def i_pad(i):
    """128-aligned padded input length (Tensor-Core block size)."""
    return int(math.ceil(i / 128.0) * 128)


try:
    from _data_4rail_override import OVERRIDE_4RAIL
except ImportError:
    OVERRIDE_4RAIL = {}


def load_rows(data_dict=None, *_unused_args, **_unused_kwargs):
    """Yield rows from the self-contained DATA_16BIT dict (default) or any
    {fw: [(I, O, ttft, tpot, dec_w, e_tok_mj), ...]} mapping.

    When OVERRIDE_4RAIL is available, every (fw, I, O) cell's `dec_w` and
    `e_tok_mj` are replaced with the 4-rail values derived directly from
    `dec_total_mw + dec_dram_mw` in the source CSV. This produces uniform
    4-rail accounting across the entire grid, regardless of which rail
    convention the original bench harness wrote into `decode_energy_mj` for
    that sweep date (short-context = 4-rail, long-context = 3-rail in
    the CSV).
    """
    if data_dict is None:
        data_dict = DATA_16BIT
    out = []
    for fw, tuples in data_dict.items():
        if fw not in FW_ORDER:
            continue
        for tup in tuples:
            I, O, ttft, tpot, dec_w, e_tok = tup
            # Apply 4-rail override if available for this cell
            ovr = OVERRIDE_4RAIL.get((fw, int(I), int(O)))
            if ovr:
                dec_w, e_tok = ovr
            out.append({
                "fw": fw,
                "I": int(I),
                "O": int(O),
                "generated": int(O),
                "ttft": ttft,
                "tpot": tpot,
                "dec_w": dec_w,
                "e_tok_mj": e_tok,
            })
    return out


def lsq(X, y):
    """numpy LSQ: X is (n, k), y is (n,). Returns coefficients vector or None."""
    if len(y) < X.shape[1]:
        return None
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def mape(measured, predicted):
    pairs = [(m, p) for m, p in zip(measured, predicted) if m]
    if not pairs:
        return None
    return 100.0 * mean(abs(p - m) / m for m, p in pairs)


# ----------------------- Prefill TTFT fit ---------------------------
def fit_prefill(rows_fw):
    """L_pref(I) = a I_pad^2 + b I_pad + c. Average TTFT at each I across gen
    values (TTFT depends on I, not O)."""
    by_I = {}
    for r in rows_fw:
        if r["ttft"] is None:
            continue
        by_I.setdefault(r["I"], []).append(r["ttft"])
    Is = sorted(by_I)
    if len(Is) < 3:
        return None
    Ip = np.array([i_pad(i) for i in Is], dtype=float)
    Tm = np.array([mean(by_I[i]) for i in Is], dtype=float)
    X = np.column_stack([Ip * Ip, Ip, np.ones_like(Ip)])
    coef = lsq(X, Tm)
    if coef is None:
        return None
    a, b, c = coef
    pred = a * Ip * Ip + b * Ip + c
    return {
        "a": a, "b": b, "c": c,
        "Is": Is, "Tm": Tm, "pred": pred,
        "mape": mape(Tm, pred),
    }


# ----------------------- Decode TPOT fit ----------------------------
def fit_decode(rows_fw):
    """TPOT(I, O) = m * (I + (O - 1) / 2) + n.
    The effective-context variable x = I + (O-1)/2 is the mean attention
    sequence length over the decode phase."""
    pts = [(r["I"] + (r["O"] - 1) / 2.0, r["tpot"]) for r in rows_fw
           if r["tpot"] is not None]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    X = np.column_stack([xs, np.ones_like(xs)])
    coef = lsq(X, ys)
    if coef is None:
        return None
    m, n = coef
    pred = m * xs + n
    return {
        "m": m, "n": n,
        "xs": xs, "ys": ys, "pred": pred,
        "mape": mape(ys, pred),
    }


# ----------------------- Decode power fit ---------------------------
def fit_power(rows_fw):
    """P_dec(O) = y ln(O) + z. EdgeReasoning Fig 5 uses this above O>=64;
    every cell in our grid satisfies O>=128 so we use the single branch."""
    by_O = {}
    for r in rows_fw:
        if r["dec_w"] is None:
            continue
        by_O.setdefault(r["O"], []).append(r["dec_w"])
    Os = sorted(by_O)
    if len(Os) < 2:
        return None
    Oa = np.array(Os, dtype=float)
    Pm = np.array([mean(by_O[o]) for o in Os], dtype=float)
    X = np.column_stack([np.log(Oa), np.ones_like(Oa)])
    coef = lsq(X, Pm)
    if coef is None:
        return None
    y, z = coef
    pred = y * np.log(Oa) + z
    return {
        "y": y, "z": z,
        "Os": Os, "Pm": Pm, "pred": pred,
        "mape": mape(Pm, pred),
    }


# ----------------------- Energy per token fit -----------------------
def fit_energy(rows_fw):
    """E_tok(O) = alpha ln(O) + beta (logarithmic, mJ/tok)."""
    by_O = {}
    for r in rows_fw:
        if r["e_tok_mj"] is None:
            continue
        by_O.setdefault(r["O"], []).append(r["e_tok_mj"])
    Os = sorted(by_O)
    if len(Os) < 2:
        return None
    Oa = np.array(Os, dtype=float)
    Em = np.array([mean(by_O[o]) for o in Os], dtype=float)
    X = np.column_stack([np.log(Oa), np.ones_like(Oa)])
    coef = lsq(X, Em)
    if coef is None:
        return None
    alpha, beta = coef
    pred = alpha * np.log(Oa) + beta
    return {
        "alpha": alpha, "beta": beta,
        "Os": Os, "Em": Em, "pred": pred,
        "mape": mape(Em, pred),
    }


# ----------------------- Plotters -----------------------------------
def _setup_axes(ax, xlabel, ylabel, title=None, logx=False, logy=False):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)


def plot_prefill(per_fw, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for fw in FW_ORDER:
        f = per_fw.get(fw, {}).get("prefill")
        if f is None:
            continue
        Ip = np.array([i_pad(i) for i in f["Is"]], dtype=float)
        ax.scatter(Ip, f["Tm"], color=FW_COLOR[fw], marker=FW_MARKER[fw],
                   label=FW_LABEL[fw], s=32, zorder=3)
        I_grid = np.linspace(Ip.min(), Ip.max(), 200)
        T_fit = f["a"] * I_grid * I_grid + f["b"] * I_grid + f["c"]
        ax.plot(I_grid, T_fit, color=FW_COLOR[fw], linestyle="--",
                linewidth=1.2, alpha=0.8, zorder=2)
    _setup_axes(ax, "Input sequence length $I$ (tokens)", "Prefill TTFT (ms)")
    ax.legend(fontsize=9, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_tpot(per_fw, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for fw in FW_ORDER:
        f = per_fw.get(fw, {}).get("decode")
        if f is None:
            continue
        order = np.argsort(f["xs"])
        ax.scatter(f["xs"], f["ys"], color=FW_COLOR[fw], marker=FW_MARKER[fw],
                   label=FW_LABEL[fw], s=24, alpha=0.85, zorder=3)
        x_grid = np.linspace(float(f["xs"].min()), float(f["xs"].max()), 200)
        y_fit = f["m"] * x_grid + f["n"]
        ax.plot(x_grid, y_fit, color=FW_COLOR[fw], linestyle="--",
                linewidth=1.2, alpha=0.8, zorder=2)
    _setup_axes(ax, "Effective context $I + (O-1)/2$ (tokens)", "TPOT (ms/tok)",
                logx=True)
    ax.legend(fontsize=9, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_power(per_fw, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for fw in FW_ORDER:
        f = per_fw.get(fw, {}).get("power")
        if f is None:
            continue
        Oa = np.array(f["Os"], dtype=float)
        ax.scatter(Oa, f["Pm"], color=FW_COLOR[fw], marker=FW_MARKER[fw],
                   label=FW_LABEL[fw], s=32, zorder=3)
        Og = np.linspace(Oa.min(), Oa.max(), 200)
        Pfit = f["y"] * np.log(Og) + f["z"]
        ax.plot(Og, Pfit, color=FW_COLOR[fw], linestyle="--",
                linewidth=1.2, alpha=0.8, zorder=2)
    _setup_axes(ax, "Output length $O$ (tokens)", "Decode power (W)", logx=True)
    ax.legend(fontsize=9, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_energy(per_fw, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for fw in FW_ORDER:
        f = per_fw.get(fw, {}).get("energy")
        if f is None:
            continue
        Oa = np.array(f["Os"], dtype=float)
        ax.scatter(Oa, f["Em"], color=FW_COLOR[fw], marker=FW_MARKER[fw],
                   label=FW_LABEL[fw], s=32, zorder=3)
        Og = np.linspace(Oa.min(), Oa.max(), 200)
        Efit = f["alpha"] * np.log(Og) + f["beta"]
        ax.plot(Og, Efit, color=FW_COLOR[fw], linestyle="--",
                linewidth=1.2, alpha=0.8, zorder=2)
    _setup_axes(ax, "Output length $O$ (tokens)", "Energy / output token (mJ)",
                logx=True)
    ax.legend(fontsize=9, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


# ----------------------- Table emitter ------------------------------
def emit_table(per_fw, quant):
    print(r"% Auto-generated by figs/scripts/fit_perf_models.py — do not hand-edit.")
    print(rf"% Llama-3.2-1B {quant}, AGX Orin 32 GB locked-clock.")
    print(r"\begin{table}[t]\centering\small")
    print(r"\begin{tabular}{lrrrrrrrrr}")
    print(r"\toprule")
    print(r"          & \multicolumn{3}{c}{Prefill $aI_p^2{+}bI_p{+}c$} & "
          r"\multicolumn{2}{c}{TPOT $mx{+}n$} & "
          r"\multicolumn{2}{c}{$P_{\rm dec}\,y\ln O{+}z$} & "
          r"\multicolumn{2}{c}{$E_{\rm tok}\,\alpha\ln O{+}\beta$} \\")
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}")
    print(r"Framework & $a$ & $b$ & $c$ & $m$ & $n$ & $y$ & $z$ & $\alpha$ & $\beta$ \\")
    print(r"\midrule")
    for fw in FW_ORDER:
        ent = per_fw.get(fw, {})
        cells = []
        pre = ent.get("prefill")
        if pre:
            cells += [f"{pre['a']:.2e}", f"{pre['b']:.2e}", f"{pre['c']:.2f}"]
        else:
            cells += ["---"] * 3
        dec = ent.get("decode")
        if dec:
            cells += [f"{dec['m']:.2e}", f"{dec['n']:.2f}"]
        else:
            cells += ["---"] * 2
        pw = ent.get("power")
        if pw:
            cells += [f"{pw['y']:.2f}", f"{pw['z']:.2f}"]
        else:
            cells += ["---"] * 2
        en = ent.get("energy")
        if en:
            cells += [f"{en['alpha']:.2f}", f"{en['beta']:.2f}"]
        else:
            cells += ["---"] * 2
        print(f"{FW_LABEL[fw]} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule\end{tabular}")
    print(r"\caption{Fitted analytical scaling models per framework on")
    print(rf"  Llama-3.2-1B {quant}, AGX Orin 32~GB locked-clock. Prefill")
    print(r"  latency $L_{\rm pref}=aI_p^2+bI_p+c$ with $I_p=\lceil I/128\rceil\cdot128$;")
    print(r"  per-token decode $\tpot=m\,(I+(O-1)/2)+n$; logarithmic decode-power")
    print(r"  and energy-per-token in $O$. Held-out MAPE per metric is reported")
    print(r"  in the text.}")
    print(r"\label{tab:perf-fits}")
    print(r"\end{table}")


def print_residuals(per_fw, file=sys.stderr):
    print("% --- fitted coefficients + MAPE per framework ---", file=file)
    for fw in FW_ORDER:
        ent = per_fw.get(fw)
        if not ent:
            continue
        print(f"% {fw}:", file=file)
        for key in ("prefill", "decode", "power", "energy"):
            f = ent.get(key)
            if not f:
                continue
            coefs = {k: v for k, v in f.items() if k not in ("Is", "Tm", "pred", "xs", "ys", "Os", "Pm", "Em")}
            coef_str = ", ".join(f"{k}={v:.4g}" for k, v in coefs.items())
            print(f"%   {key:<8s} {coef_str}", file=file)


def _load_data_dict_from_csv(csv_path, model_filter="Llama-3.2-1B"):
    """Read sweep_locked{,_orin_15run}.csv, produce a DATA_16BIT-shape dict:
        {fw: [(I, O, ttft_ms, tpot_ms, dec_w, e_tok_mj), ...], ...}

    Filters to a single model (default Llama-3.2-1B, matches paper's DATA_16BIT)
    and one 16-bit quantization per framework. Uses 4-rail power sum for
    dec_w = (dec_gpu_mw + dec_cpu_mw + dec_soc_mw + dec_dram_mw)/1000.
    e_tok_mj = decode_energy_mj / gen_tokens.

    QUANT_16BIT map: {trtllm: fp16, llamacpp: f16, vllm: fp16, sglang: fp16,
    pytorch: bf16}. Cells duplicated per (I, O) — first match wins.
    """
    import csv as _csv
    QUANT_16BIT = {"trtllm": "fp16", "llamacpp": "f16",
                   "vllm": "fp16", "sglang": "fp16", "pytorch": "bf16"}
    seen_cells = {}  # (fw, I, O) -> tuple  (first match wins to avoid dupes)
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            fw = r.get("framework", "")
            if fw not in QUANT_16BIT:
                continue
            if r.get("quantization", "") != QUANT_16BIT[fw]:
                continue
            if r.get("model", "") != model_filter:
                continue
            try:
                I = int(r["prompt_tokens"])
                O = int(r["gen_tokens"])
                key = (fw, I, O)
                if key in seen_cells:
                    continue
                ttft = float(r["ttft_ms"])
                tpot = float(r["tpot_ms"])
                gpu = float(r.get("dec_gpu_mw") or 0)
                cpu = float(r.get("dec_cpu_mw") or 0)
                soc = float(r.get("dec_soc_mw") or 0)
                dram = float(r.get("dec_dram_mw") or 0)
                dec_w = (gpu + cpu + soc + dram) / 1000.0
                gen_toks = int(r.get("generated_tokens") or O)
                e_tok = float(r.get("decode_energy_mj") or 0) / max(gen_toks, 1)
            except (ValueError, KeyError, TypeError):
                continue
            seen_cells[key] = (I, O, ttft, tpot, dec_w, e_tok)

    out = {}
    for (fw, I, O), tup in seen_cells.items():
        out.setdefault(fw, []).append(tup)
    for fw in out:
        out[fw].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama-3.2-1B",
                    help="(documentation only — used for the emit_table label)")
    ap.add_argument("--quant", default="16-bit",
                    help="Quantization label for the emit_table header (default 16-bit)")
    ap.add_argument("--csv", default=None,
                    help="Explicit path to sweep_locked{,_orin_15run}.csv; "
                         "if omitted, auto-locates from artifact/data/chat/")
    args = ap.parse_args()

    # Locate CSV; prefer 15-run aggregate. Falls back to hardcoded DATA_16BIT
    # if neither exists (submission-time reproducibility).
    csv_path = None
    if args.csv:
        csv_path = Path(args.csv)
    else:
        _ART = Path(__file__).resolve().parent.parent.parent  # scripts/plot -> scripts -> artifact
        for cand in (_ART / "data/chat/sweep_locked.csv",):
            if cand.is_file():
                csv_path = cand
                break

    if csv_path and csv_path.is_file():
        print(f"[gen_fig08_energy_fits] loading data from {csv_path}",
              file=sys.stderr)
        data_dict = _load_data_dict_from_csv(csv_path)
        if not data_dict:
            print(f"  no matching rows; falling back to hardcoded DATA_16BIT",
                  file=sys.stderr)
            data_dict = DATA_16BIT
    else:
        print(f"[gen_fig08_energy_fits] no CSV found; using hardcoded DATA_16BIT",
              file=sys.stderr)
        data_dict = DATA_16BIT

    rows = load_rows(data_dict)
    if not rows:
        print("no rows loaded", file=sys.stderr)
        sys.exit(1)

    by_fw = {}
    for r in rows:
        by_fw.setdefault(r["fw"], []).append(r)

    per_fw = {}
    for fw, rs in by_fw.items():
        per_fw[fw] = {
            "prefill": fit_prefill(rs),
            "decode":  fit_decode(rs),
            "power":   fit_power(rs),
            "energy":  fit_energy(rs),
        }

    print_residuals(per_fw)

    # Fig 8 output = energy fits. TTFT/TPOT/power fits are Thor-owned
    # (cross-platform); we still emit them here for local sanity but the paper
    # picks them up from Thor's generators.
    plot_prefill(per_fw, OUT_DIR / "fig_v5_ttft_fits")
    plot_tpot(per_fw,    OUT_DIR / "fig_v5_tpot_fits")
    plot_power(per_fw,   OUT_DIR / "fig_v5_power_fits")
    plot_energy(per_fw,  OUT_DIR / "fig_v5_energy_fits")

    emit_table(per_fw, args.quant)


if __name__ == "__main__":
    main()
