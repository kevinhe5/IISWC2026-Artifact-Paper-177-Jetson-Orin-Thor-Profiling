#!/usr/bin/env bash
# ============================================================================
# grant_sudo.sh — one-time sudoers setup for the measurement pipeline.
#
#   sudo bash scripts/collect/orin/grant_sudo.sh          # grants your user
#   sudo bash scripts/collect/orin/grant_sudo.sh alice    # or a named user
#
# Installs /etc/sudoers.d/orin-artifact whitelisting, passwordless, EXACTLY
# the five privileged commands the pipeline invokes (nothing broader):
#   * jetson_clocks                      — lock GPU/EMC clocks (preflight)
#   * nvpmodel -q / nvpmodel -m 0        — query/set MAXN power mode
#   * sysctl vm.swappiness=10            — cap swappiness for the runs
#   * sh -c 'echo 3 > /proc/sys/vm/drop_caches'
#                                        — page-cache eviction between runs
#   * cat /sys/kernel/debug/bpmp/debug/clk/emc/rate
#                                        — verify the locked EMC frequency
#
# The file is syntax-validated with `visudo -cf` BEFORE installation, so a
# parse error can never lock you out of sudo. Undo at any time with:
#   sudo rm /etc/sudoers.d/orin-artifact
#
# Alternative to this script: run the collection under `sudo -E` — but then
# everything written under PROFILE_ROOT and the HF cache in your home ends
# up root-owned, which breaks later non-root runs. Prefer this whitelist.
# ============================================================================
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root:  sudo bash $0" >&2
    exit 1
fi

TARGET_USER="${1:-${SUDO_USER:-}}"
if [ -z "${TARGET_USER}" ] || [ "${TARGET_USER}" = "root" ]; then
    echo "Cannot determine the non-root user to grant. Pass it explicitly:" >&2
    echo "  sudo bash $0 <username>" >&2
    exit 1
fi
if ! id "${TARGET_USER}" >/dev/null 2>&1; then
    echo "No such user: ${TARGET_USER}" >&2
    exit 1
fi

DEST=/etc/sudoers.d/orin-artifact
TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT

cat > "${TMP}" <<EOF
# Installed by scripts/collect/orin/grant_sudo.sh (IISWC 2026 artifact).
# Grants ${TARGET_USER} passwordless sudo for ONLY the five commands the
# measurement pipeline runs. Remove with: sudo rm ${DEST}
${TARGET_USER} ALL=(ALL) NOPASSWD: /usr/bin/jetson_clocks
${TARGET_USER} ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel -q, /usr/sbin/nvpmodel -m 0, /usr/bin/nvpmodel -q, /usr/bin/nvpmodel -m 0
${TARGET_USER} ALL=(ALL) NOPASSWD: /usr/sbin/sysctl vm.swappiness=10
${TARGET_USER} ALL=(ALL) NOPASSWD: /usr/bin/cat /sys/kernel/debug/bpmp/debug/clk/emc/rate
${TARGET_USER} ALL=(ALL) NOPASSWD: /usr/bin/sh -c echo 3 > /proc/sys/vm/drop_caches, /bin/sh -c echo 3 > /proc/sys/vm/drop_caches
EOF

# Refuse to install anything visudo won't parse — a broken sudoers.d file
# can disable sudo system-wide.
visudo -cf "${TMP}"
install -m 0440 -o root -g root "${TMP}" "${DEST}"

echo "Installed ${DEST} for user ${TARGET_USER}."
echo "Verify (as ${TARGET_USER}):"
echo "  sudo -n /usr/bin/jetson_clocks \\"
echo "    && sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' \\"
echo "    && echo SUDO-SETUP-OK"
