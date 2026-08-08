#!/usr/bin/env bash
# Build and run the buffer-size triad bandwidth sweep on the local Jetson.
# Auto-detects platform (Orin/Thor) to set the spec peak for %-of-peak.
#
# Usage:  ./run_triad_sweep.sh [extra args forwarded to the binary]
#   e.g.  ./run_triad_sweep.sh --max-mb 4096 --trials 9
#
# Lock clocks first for a deterministic result:
#   sudo nvpmodel -m 0 && sudo jetson_clocks
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- locate nvcc (often not on PATH on Jetson) ---
NVCC="$(command -v nvcc || true)"
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
if [ -z "$NVCC" ]; then
  echo "ERROR: nvcc not found. Run inside a CUDA container or add /usr/local/cuda/bin to PATH." >&2
  exit 1
fi
echo "# nvcc: $NVCC ($($NVCC --version | grep -oE 'release [0-9.]+' | head -1))" >&2

# --- detect platform + spec peak ---
MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
PEAK=""
case "$MODEL" in
  *Thor*) PEAK=273.0 ;;   # AGX Thor  LPDDR5X spec
  *Orin*) PEAK=204.8 ;;   # AGX Orin  LPDDR5  spec (32/64 GB)
esac
echo "# model: $MODEL  ->  spec peak: ${PEAK:-auto}" >&2

# --- clock-lock sanity check (warn only) ---
if command -v nvpmodel >/dev/null 2>&1; then
  echo "# nvpmodel: $(nvpmodel -q 2>/dev/null | grep -i 'NV Power Mode' || true)" >&2
fi
echo "# NOTE: ensure 'sudo jetson_clocks' has been run, else results underread." >&2

# --- build (targets the present GPU; no need to hardcode sm_87/sm_110) ---
BIN="$HERE/triad_bw_sweep"
echo "# building..." >&2
"$NVCC" -O3 -arch=native -o "$BIN" triad_bw_sweep.cu

# --- run: human-readable to console, CSV to a timestamped file ---
STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="$(echo "$MODEL" | grep -oiE 'orin|thor' | head -1 | tr 'A-Z' 'a-z' || echo dev)"
CSV="triad_sweep_${TAG}_${STAMP}.csv"

PEAK_ARG=()
[ -n "$PEAK" ] && PEAK_ARG=(--peak "$PEAK")

echo "# ---- human-readable ----" >&2
"$BIN" "${PEAK_ARG[@]}" "$@"
echo "# ---- writing CSV: $CSV ----" >&2
"$BIN" "${PEAK_ARG[@]}" --csv "$@" | tee "$CSV"
echo "# done -> $HERE/$CSV" >&2
