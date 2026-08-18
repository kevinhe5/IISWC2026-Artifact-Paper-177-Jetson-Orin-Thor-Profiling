#!/usr/bin/env bash
# ============================================================================
# preflight.sh — Orin Workflow B environment sanity check + lock.
#
# Referenced by scripts/collect/orin/sweep/sweep.sh (error path).
# Run once before the first collection; safe to re-run between stages.
#
# Checks / actions (each is independent; the script does NOT exit on a
# single WARN so you see all issues at once):
#   1. swap in use < 5 GB (hard fail otherwise; suggests reboot or swapoff)
#   2. vm.swappiness ≤ 10 (sets it if higher; needs sudo)
#   3. Docker daemon reachable + no lingering containers
#   4. jetson_clocks locked (GPU 1300.5 MHz, EMC 3199 MHz, MAXN power mode)
#   5. drop_caches so cold-start latencies are consistent
#   6. Profile root exists ($PROFILE_ROOT/models, $PROFILE_ROOT/benchmarks)
#
# The script uses `sudo -n` (non-interactive). If a sudoer entry for
# `jetson_clocks`, `nvpmodel`, `sysctl`, `sh -c 'echo 3 > .../drop_caches'`
# is not present, run the whole preflight once via `sudo bash preflight.sh`.
# ============================================================================
set -uo pipefail

PROFILE_ROOT="${PROFILE_ROOT:-${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/profile}}"
DATA_ROOT="${PROFILE_ROOT}"   # alias for legacy sub-scripts
GPU_DEVFREQ="/sys/class/devfreq/17000000.gpu"

WARNINGS=0
FAILS=0
banner() { echo; echo "[preflight] $*"; }
warn()   { echo "  WARN: $*"; WARNINGS=$((WARNINGS+1)); }
fail()   { echo "  FAIL: $*"; FAILS=$((FAILS+1)); }
ok()     { echo "  OK:   $*"; }
# Run a command with root privilege: directly when already root, else via
# sudo -n. Tried PER COMMAND (not gated on blanket sudo) so that per-command
# NOPASSWD whitelists — grant_sudo.sh's or a site's own — are honored.
priv()   { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo -n "$@"; fi; }

# ---- Are we already root? if not, is sudo -n available? -------------------
# NOTE: `sudo -n true` only passes with BLANKET passwordless sudo. The
# grant_sudo.sh whitelist grants specific commands, so also probe one of
# the actually-whitelisted commands (jetson_clocks --show is read-only).
have_sudo=0
if [ "$(id -u)" -eq 0 ]; then
    have_sudo=1
    SUDO=""
elif sudo -n true 2>/dev/null; then
    have_sudo=1
    SUDO="sudo -n"
elif sudo -n /usr/bin/jetson_clocks --show >/dev/null 2>&1; then
    have_sudo=1
    SUDO="sudo -n"
else
    have_sudo=0
    SUDO=""
fi
if [ "$have_sudo" -eq 0 ]; then
    banner "sudo"
    warn "sudo -n not available. Steps 2/4/5 will be skipped. Re-run as:"
    echo "        sudo bash scripts/collect/orin/preflight.sh"
fi

# ---- 1. Swap --------------------------------------------------------------
banner "swap check"
SWAP_USED_KB=$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)
if [ "$SWAP_USED_KB" -gt 5242880 ]; then
    echo "  ERROR: swap in use ${SWAP_USED_KB} KB (> 5 GB). Free it first:"
    echo "    sudo swapoff -a && sudo swapon -a"
    echo "  (or reboot). Then re-run preflight.sh."
    exit 1
fi
ok "swap in use ${SWAP_USED_KB} KB (< 5 GB)"

# ---- 2. Swappiness --------------------------------------------------------
banner "vm.swappiness"
SWAPPINESS=$(cat /proc/sys/vm/swappiness)
if [ "$SWAPPINESS" -gt 10 ]; then
    if priv sysctl vm.swappiness=10 >/dev/null 2>&1; then
        ok "set vm.swappiness=10"
    else
        warn "vm.swappiness=${SWAPPINESS}; want ≤ 10 but cannot set (no sudo grant)"
    fi
else
    ok "vm.swappiness=${SWAPPINESS}"
fi

# ---- 3. Docker ------------------------------------------------------------
banner "docker daemon"
if ! docker info >/dev/null 2>&1; then
    echo "  ERROR: docker daemon not reachable. Start it: sudo systemctl start docker"
    exit 1
fi
RUNNING=$(docker ps -q | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    warn "$RUNNING container(s) already running (sweep needs an idle GPU):"
    docker ps --format '        {{.ID}} {{.Image}}'
    if [ "$have_sudo" -eq 1 ]; then
        docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 && ok "cleaned up"
    fi
else
    ok "docker up, no containers running"
fi

# ---- 4. jetson_clocks + power mode ---------------------------------------
banner "jetson_clocks lock"
if priv /usr/bin/jetson_clocks 2>/dev/null; then
    ok "jetson_clocks invoked"
else
    warn "cannot invoke jetson_clocks (no sudo grant) — verifying current clock state instead"
fi
NV_MODE=$(priv nvpmodel -q 2>/dev/null | awk '/NV Power Mode/{getline; print}')
if [ -z "$NV_MODE" ]; then
    warn "cannot query nvpmodel (no sudo grant) — power mode unverified"
elif [ "$NV_MODE" != "MAXN" ]; then
    if priv nvpmodel -m 0 >/dev/null 2>&1; then
        ok "NV Power Mode → MAXN"
    else
        fail "NV Power Mode is '${NV_MODE}' and cannot set MAXN — run once: sudo nvpmodel -m 0"
    fi
else
    ok "NV Power Mode: MAXN"
fi

# Read current GPU / EMC freq (no sudo needed for GPU devfreq; EMC needs sudo)
GPU_FREQ=$(cat "${GPU_DEVFREQ}/cur_freq" 2>/dev/null || echo unknown)
GPU_MAX=$(cat "${GPU_DEVFREQ}/max_freq" 2>/dev/null || echo unknown)
if [ "$GPU_FREQ" = "$GPU_MAX" ] && [ "$GPU_FREQ" != "unknown" ]; then
    ok "GPU freq ${GPU_FREQ} Hz (= max, locked)"
else
    fail "GPU freq ${GPU_FREQ} Hz (max ${GPU_MAX} Hz) — clocks NOT locked; run once: sudo jetson_clocks"
fi
EMC_FREQ=$(priv cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>/dev/null || echo unknown)
if [ "$EMC_FREQ" = "unknown" ]; then
    warn "EMC freq unreadable (needs sudo) — assuming locked with GPU"
elif [ "$EMC_FREQ" != "3199000000" ]; then
    fail "EMC freq ${EMC_FREQ} Hz != locked 3199000000 — run once: sudo jetson_clocks"
else
    ok "EMC freq ${EMC_FREQ} Hz  (expected 3199000000)"
fi

# ---- 5. drop caches -------------------------------------------------------
banner "drop caches"
sync
if priv sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    ok "drop_caches=3"
else
    warn "cannot drop caches (no sudo grant) — page-cache pressure may add variance"
fi

# ---- 6. DATA_ROOT layout --------------------------------------------------
banner "DATA_ROOT layout"
echo "  PROFILE_ROOT=${PROFILE_ROOT}"
for sub in models benchmarks; do
    if [ ! -d "${PROFILE_ROOT}/${sub}" ]; then
        fail "${PROFILE_ROOT}/${sub} missing — run: bash scripts/collect/orin/prepare_orin.sh"
    else
        ok "${PROFILE_ROOT}/${sub}"
    fi
done

# ---- Summary --------------------------------------------------------------
banner "preflight DONE — ${FAILS} failure(s), ${WARNINGS} warning(s)"
if [ "$FAILS" -gt 0 ]; then
    echo "  FAILs are measurement-critical (unlocked clocks / wrong power mode /"
    echo "  missing data layout) — fix them before running the sweep."
    exit 2
fi
if [ "$WARNINGS" -gt 0 ]; then
    echo "  Proceeding despite warnings: hygiene extras (cache drop, swappiness,"
    echo "  EMC verification) are unavailable without a sudo grant. Results may"
    echo "  carry slightly more run-to-run variance."
fi
