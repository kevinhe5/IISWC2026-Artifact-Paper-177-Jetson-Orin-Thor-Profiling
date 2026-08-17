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
banner() { echo; echo "[preflight] $*"; }
warn()   { echo "  WARN: $*"; WARNINGS=$((WARNINGS+1)); }
ok()     { echo "  OK:   $*"; }

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
    if [ "$have_sudo" -eq 1 ]; then
        $SUDO sysctl vm.swappiness=10 >/dev/null && ok "set vm.swappiness=10" \
                                                 || warn "sysctl failed (permissions?)"
    else
        warn "vm.swappiness=${SWAPPINESS}; want ≤ 10. Need sudo."
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
if [ "$have_sudo" -eq 1 ]; then
    $SUDO /usr/bin/jetson_clocks 2>/dev/null && ok "jetson_clocks invoked" \
                                             || warn "jetson_clocks failed"
    # Set MAXN (mode 0) if not already
    NV_MODE=$($SUDO nvpmodel -q 2>/dev/null | awk '/NV Power Mode/{getline; print}')
    if [ "$NV_MODE" != "MAXN" ]; then
        $SUDO nvpmodel -m 0 >/dev/null && ok "NV Power Mode → MAXN" \
                                                || warn "nvpmodel failed"
    else
        ok "NV Power Mode: MAXN"
    fi
else
    warn "cannot invoke jetson_clocks / nvpmodel — need sudo"
fi

# Read current GPU / EMC freq (no sudo needed for GPU devfreq; EMC needs sudo)
GPU_FREQ=$(cat "${GPU_DEVFREQ}/cur_freq" 2>/dev/null || echo unknown)
GPU_MAX=$(cat "${GPU_DEVFREQ}/max_freq" 2>/dev/null || echo unknown)
if [ "$GPU_FREQ" = "$GPU_MAX" ] && [ "$GPU_FREQ" != "unknown" ]; then
    ok "GPU freq ${GPU_FREQ} Hz (= max, locked)"
else
    warn "GPU freq ${GPU_FREQ} Hz (max ${GPU_MAX} Hz) — clocks may not be locked"
fi
EMC_FREQ=$($SUDO cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>/dev/null || echo unknown)
if [ "$EMC_FREQ" != "unknown" ]; then
    ok "EMC freq ${EMC_FREQ} Hz  (expected 3199000000)"
else
    warn "EMC freq unreadable (needs sudo)"
fi

# ---- 5. drop caches -------------------------------------------------------
banner "drop caches"
sync
if [ "$have_sudo" -eq 1 ]; then
    $SUDO sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null && ok "drop_caches=3" \
                                                              || warn "drop_caches failed"
else
    warn "cannot drop caches — need sudo"
fi

# ---- 6. DATA_ROOT layout --------------------------------------------------
banner "DATA_ROOT layout"
echo "  PROFILE_ROOT=${PROFILE_ROOT}"
for sub in models benchmarks; do
    if [ ! -d "${PROFILE_ROOT}/${sub}" ]; then
        warn "${PROFILE_ROOT}/${sub} missing — run: bash scripts/collect/orin/prepare_orin.sh"
    else
        ok "${PROFILE_ROOT}/${sub}"
    fi
done

# ---- Summary --------------------------------------------------------------
banner "preflight DONE — ${WARNINGS} warning(s)"
if [ "$WARNINGS" -gt 0 ]; then
    echo "  Address the WARNs above before running the sweep."
    exit 2
fi
