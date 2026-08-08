#!/usr/bin/env bash
# Generic SWE-bench-B repo prep — reads the active task JSONL, groups by
# base image, pulls each base ONCE, then materialises one dir per bug
# variant via cp + git checkout. Replaces the prior hard-coded 5-task
# version.
#
# Inputs (env override-able):
#   TASKS_JSONL : path to the task list (default = swebench_live_30tasks.jsonl)
#   ORG_NAME_DH : Docker Hub namespace (default `swebench` — SWE-smith's)
#   KEEP_IMAGES : 1 to retain pulled images (default 0 — saves ~3-5 GB)
#
# instance_id format: `<owner>__<repo>.<sha>.<bug_suffix>`. The base image
# name from SWE-smith's `swesmith.profiles.base.RepoProfile.image_name` is
#   {ORG_NAME_DH}/swesmith.x86_64.<owner>_1776_<repo>.<sha[:8]>
# (slash → `_1776_` because Docker image names can't contain `/`).
#
# Each variant is on a git branch named after its full instance_id, so we
# pull the base, extract /testbed (with .git intact), then on the host do
# `git checkout <instance_id>` to materialise the buggy state.

set -e

BENCH_DIR="${PROFILE_ROOT:-/nvme/iiswc/Jetson_profile}/data/benchmarks/orin"
REPOS_DIR="${BENCH_DIR}/swebench_repos"
TASKS_JSONL="${TASKS_JSONL:-${BENCH_DIR}/data/cache_workloads/swebench_live_30tasks.jsonl}"
ORG_NAME_DH="${ORG_NAME_DH:-swebench}"
KEEP_IMAGES="${KEEP_IMAGES:-0}"

[ ! -f "${TASKS_JSONL}" ] && { echo "ERROR: ${TASKS_JSONL} missing" >&2; exit 1; }

mkdir -p "${REPOS_DIR}"

# Parse jsonl → list of (instance_id, base_key). base_key derives the
# image tag suffix: owner_1776_repo.sha8 (sha shortened to 8 chars).
mapfile -t INSTANCES < <(python3 - <<'PY'
import json, sys, os
src = os.environ["TASKS_JSONL"]
seen = set()
for line in open(src):
    d = json.loads(line)
    iid = d.get("instance_id")
    if not iid or iid in seen: continue
    seen.add(iid)
    parts = iid.split(".")
    if len(parts) < 2:
        continue
    owner_repo = parts[0]      # python__mypy
    sha = parts[1][:8]         # e93f06ce
    if "__" not in owner_repo:
        continue
    owner, repo = owner_repo.split("__", 1)
    base = f"{owner}_1776_{repo}.{sha}".lower()
    print(f"{iid}\t{base}")
PY
)

# Group by base — same base image serves multiple bug variants.
declare -A BASE_INSTANCES
declare -A BASE_PULLED   # base → "" or "yes" once pulled+extracted-template
TEMPLATE_DIR=""
for line in "${INSTANCES[@]}"; do
    IFS=$'\t' read -r iid base <<<"$line"
    BASE_INSTANCES[$base]+="${iid} "
done

echo "=== SWE-bench-B repo prep (generic) ==="
echo "  tasks_jsonl : ${TASKS_JSONL}"
echo "  registry    : ${ORG_NAME_DH}/swesmith.x86_64.<base>"
echo "  target      : ${REPOS_DIR}/"
echo "  instances   : ${#INSTANCES[@]} across ${#BASE_INSTANCES[@]} base repos"
echo

for base in "${!BASE_INSTANCES[@]}"; do
    IMG="${ORG_NAME_DH}/swesmith.x86_64.${base}:latest"
    iids="${BASE_INSTANCES[$base]}"
    first_iid=$(echo $iids | awk '{print $1}')
    TEMPLATE_DIR="${REPOS_DIR}/_template_${base}"

    # Skip pulling if ALL variants under this base already exist
    all_present=1
    for iid in $iids; do
        if [ ! -d "${REPOS_DIR}/${iid}/.git" ]; then
            all_present=0; break
        fi
    done
    if [ "${all_present}" = "1" ]; then
        echo "[skip base ${base}] all $(echo $iids | wc -w) variant(s) already extracted"
        continue
    fi

    echo "[base ${base}]"
    echo "  pull   : ${IMG}"
    if ! docker pull --platform linux/amd64 "${IMG}" >/dev/null 2>&1; then
        echo "  ERROR: pull failed — ${IMG}"
        continue
    fi

    # Extract once to a template dir
    CID=$(docker create --platform linux/amd64 "${IMG}" /bin/sh)
    rm -rf "${TEMPLATE_DIR}"; mkdir -p "${TEMPLATE_DIR}"
    docker cp "${CID}:/testbed/." "${TEMPLATE_DIR}/"
    docker rm "${CID}" >/dev/null
    find "${TEMPLATE_DIR}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
    find "${TEMPLATE_DIR}" -type f -name '*.pyc' -delete 2>/dev/null || true

    # Materialise each bug variant by copy + git checkout
    for iid in $iids; do
        DEST="${REPOS_DIR}/${iid}"
        if [ -d "${DEST}/.git" ]; then
            cur=$(git -C "${DEST}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
            if [ "$cur" = "$iid" ]; then
                echo "  [skip variant ${iid}] already on branch"
                continue
            fi
        fi
        rm -rf "${DEST}"
        cp -a "${TEMPLATE_DIR}" "${DEST}"
        if git -C "${DEST}" checkout "${iid}" >/dev/null 2>&1; then
            HEAD_BR=$(git -C "${DEST}" rev-parse --abbrev-ref HEAD)
            sz=$(du -sh "${DEST}" 2>/dev/null | cut -f1)
            echo "  [ok variant] ${iid}  (${sz}, branch=${HEAD_BR})"
        else
            echo "  [WARN] ${iid} branch not found in base — available:"
            git -C "${DEST}" branch -a 2>&1 | grep "${iid:0:30}" | head -3 | sed 's/^/    /'
        fi
    done

    rm -rf "${TEMPLATE_DIR}"

    if [ "${KEEP_IMAGES}" != "1" ]; then
        docker rmi "${IMG}" >/dev/null 2>&1 || true
    fi
    echo
done

echo "=== summary ==="
echo "  $(ls -d ${REPOS_DIR}/*/ 2>/dev/null | wc -l) repo dirs extracted"
du -sh "${REPOS_DIR}" 2>/dev/null | sed 's/^/  total: /'
echo
echo "All variants ready at ${REPOS_DIR}/"
