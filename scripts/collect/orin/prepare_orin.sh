#!/usr/bin/env bash
# ============================================================================
# prepare_orin.sh — one-shot environment setup for Orin Workflow B.
#
# Performs everything you need before run_orin_collection.sh:
#   1. docker pulls for the 3 upstream framework containers
#   2. locally-built container recipes (sglang-orin, bitsandbytes-bench)
#   3. HF snapshot downloads (Llama-3.2-1B-Instruct, Qwen3-4B) — Llama is
#      GATED, so HF_TOKEN must be exported before running (see step 3 below).
#   4. GGUF quant downloads for llama.cpp + vLLM GGUF cells
#   5. TRT-LLM engine builds (fp16, int8, int4) — ~1 h wall-clock
#
# Runs everything sequentially — safe to re-run (each step skips if output
# already exists). Wall-clock on a fresh Orin: ~2-3 h dominated by TRT-LLM
# engine builds + GGUF downloads.
#
# REQUIRED before running:
#   * docker daemon reachable, at least 60 GB free on PROFILE_ROOT (default: <repo>/profile/)
#   * huggingface-cli installed  (pip install -U 'huggingface_hub[cli]')
#   * export HF_TOKEN=hf_...     — read token from
#       https://huggingface.co/settings/tokens, with the Meta Llama license
#       accepted at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct.
#       The script exits at step 3 with a [FATAL] message if this is unset.
# ============================================================================
set -uo pipefail

# Data root: PROFILE_ROOT (or legacy DATA_ROOT). Defaults to <repo>/profile/
# (gitignored) — export PROFILE_ROOT to place the ~60 GB of models/outputs
# on a different filesystem.
_REPO_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/profile"
PROFILE_ROOT="${PROFILE_ROOT:-${DATA_ROOT:-${_REPO_DEFAULT}}}"
echo "PROFILE_ROOT = ${PROFILE_ROOT}"
DATA_ROOT="${PROFILE_ROOT}"   # alias for legacy sub-scripts
HF_DIR="${PROFILE_ROOT}/models/hf_full"
GGUF_DIR="${PROFILE_ROOT}/models/gguf"
ENGINE_DIR="${PROFILE_ROOT}/models/trtllm_engines"
BENCH_DIR="${PROFILE_ROOT}/benchmarks"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "${HF_DIR}" "${GGUF_DIR}" "${ENGINE_DIR}" "${BENCH_DIR}"

banner(){ echo; echo "######## $* ######## $(date -u +%FT%TZ)"; }

# ----------------------------------------------------------------------------
# 0. Disk-space preflight (Bug 26). A full prepare lands ~35 GB under
#    PROFILE_ROOT (HF snapshots + GGUFs + 3 engines + fp16 checkpoint) plus
#    ~25 GB of docker images/build scratch. A run that hits ENOSPC midway
#    leaves partial engines that the existence-based skip would then treat
#    as complete, so refuse to start below 60 GB free.
# ----------------------------------------------------------------------------
free_gb(){ df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
NEED_GB=60
AVAIL_GB="$(free_gb "${PROFILE_ROOT}")"
DOCKER_DIR="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
DOCKER_AVAIL_GB="$(free_gb "${DOCKER_DIR}")"
if [ -n "${AVAIL_GB}" ] && [ "${AVAIL_GB}" -lt "${NEED_GB}" ]; then
    echo "[FATAL] Only ${AVAIL_GB} GB free on ${PROFILE_ROOT} — need ≥ ${NEED_GB} GB." >&2
    echo "        Free space (or point PROFILE_ROOT elsewhere) and re-run." >&2
    exit 5
fi
if [ -n "${DOCKER_AVAIL_GB}" ] && [ "${DOCKER_AVAIL_GB}" -lt 25 ]; then
    echo "[FATAL] Only ${DOCKER_AVAIL_GB} GB free on docker root ${DOCKER_DIR} — need ≥ 25 GB for images." >&2
    exit 5
fi
echo "disk preflight: ${AVAIL_GB} GB free on ${PROFILE_ROOT}, ${DOCKER_AVAIL_GB} GB on ${DOCKER_DIR} — OK"

# ----------------------------------------------------------------------------
# 0b. Install the benchmark harness into ${BENCH_DIR}. The repo keeps one
#     deduplicated copy under scripts/collect/harness/; the containers need
#     the flattened per-profiler layout, so this assembly step is REQUIRED —
#     without it every sweep cell fails with ModuleNotFoundError.
# ----------------------------------------------------------------------------
banner "0b — install benchmark harness"
if ! PROFILE_ROOT="${PROFILE_ROOT}" bash "${HERE}/install_harness.sh"; then
    echo "[FATAL] harness install failed — benchmarks would be unrunnable." >&2
    exit 6
fi

# ----------------------------------------------------------------------------
# 1. Upstream containers (dustynv images from jetson-containers)
# ----------------------------------------------------------------------------
banner "1/5 — pulling upstream containers"
for img in \
    dustynv/vllm:0.8.6-r36.4-cu128-24.04 \
    dustynv/tensorrt_llm:0.12-r36.4.0 \
    dustynv/llama_cpp:b5283-r36.4-cu128-24.04
do
    if docker image inspect "$img" >/dev/null 2>&1; then
        echo "  cached: $img"
    else
        docker pull "$img"
    fi
done

# ----------------------------------------------------------------------------
# 2. Locally-built containers
#    sglang-orin:0.4.6-sm87  — SGLang built for Ampere sm_87 (Orin GPU arch)
#    bitsandbytes-bench:r36.4.0 — PyTorch + transformers 4.57 + bnb 0.45.4.dev
# ----------------------------------------------------------------------------
banner "2/5 — locally-built containers"
# HARD FAIL on docker build failure — without these two containers, the sweep
# is missing SGLang (Fig 5/6/11/13 columns) and PyTorch (bitsandbytes int8 /
# NF4, Fig 3-11 columns). Silently proceeding produces a partial sweep whose
# CSV can't be compared to the paper, so any build error stops prepare_orin.sh.
build_or_die() {
    local tag="$1" dockerfile="$2"
    if docker image inspect "$tag" >/dev/null 2>&1; then
        echo "  cached: ${tag}"
        return 0
    fi
    echo "  Building ${tag} from ${dockerfile}"
    if ! docker build -t "${tag}" -f "${dockerfile}" "${HERE}/dockerfiles"; then
        echo
        echo "  ERROR: docker build failed for ${tag}."
        echo "         Check the log for the failing RUN stage and its exit code."
        echo "         Common causes:"
        echo "          - pip index unreachable (base image should use pypi.org — see Dockerfile ENV)"
        echo "          - base image tag drift on Docker Hub"
        echo "         Fix, then re-run: bash prepare_orin.sh"
        exit 2
    fi
}

build_or_die "sglang-orin:0.4.6-sm87"        "${HERE}/dockerfiles/Dockerfile.sglang-orin"
build_or_die "bitsandbytes-bench:r36.4.0"    "${HERE}/dockerfiles/Dockerfile.bitsandbytes-bench"

# ----------------------------------------------------------------------------
# 2b. Verify container internal versions match manifests/framework_versions.csv.
#     The paper's numbers were collected against these exact library versions;
#     the guard makes any drift immediately obvious instead of silently biasing
#     the sweep. Prints WARN and returns non-zero only if a check hard-mismatches;
#     minor patch-level drift (e.g. torch 2.5.0 vs 2.5.1) prints INFO.
# ----------------------------------------------------------------------------
banner "2b/5 — verify container framework versions vs manifest"
# HARD FAIL semantics: FAIL is fatal, WARN is a soft mismatch we surface but
# tolerate. FAIL fires when the version cannot be detected at all (container
# missing, or the library isn't importable — sweep would crash later); WARN
# fires only on version drift (importable but wrong version — sweep would run
# and might land close to paper numbers but reviewer should know).
VERIFY_FAILS=0
verify_ver() {
    local img="$1" pkg="$2" want="$3" cmd="$4"
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "  FAIL ${img##*/}: image not present — earlier build/pull step failed"
        VERIFY_FAILS=$((VERIFY_FAILS+1))
        return 1
    fi
    local got
    got=$(docker run --rm --runtime nvidia "$img" sh -c "$cmd" 2>/dev/null \
          | grep -E "^${pkg}[[:space:]]" | awk '{print $2}' | head -1)
    if [ -z "$got" ]; then
        echo "  FAIL ${img##*/}: cannot import ${pkg} — earlier build stage produced a broken image"
        VERIFY_FAILS=$((VERIFY_FAILS+1))
        return 1
    fi
    if [ "$got" = "$want" ] || [ "${got%.*}" = "${want%.*}" ]; then
        echo "  OK   ${img##*/}: ${pkg} ${got}  (expected ${want})"
    else
        echo "  WARN ${img##*/}: ${pkg} ${got}  ≠ manifest ${want}"
    fi
}

verify_ver "dustynv/vllm:0.8.6-r36.4-cu128-24.04" \
    "vllm" "0.8.6" \
    'python3 -c "import vllm; print(f\"vllm {vllm.__version__}\")"'
verify_ver "dustynv/tensorrt_llm:0.12-r36.4.0" \
    "trtllm" "0.12.0" \
    'python3 -c "import tensorrt_llm; print(f\"trtllm {tensorrt_llm.__version__}\")"'
verify_ver "sglang-orin:0.4.6-sm87" \
    "sglang" "0.4.6.post2" \
    'python3 -c "import sglang; print(f\"sglang {sglang.__version__}\")"'
# The srt ENGINE import (what the harness actually uses) only resolves its
# CUDA-gated imports (sgl_kernel, triton) with a GPU present, so it must be
# verified here with --runtime nvidia — a plain `import sglang` passes even
# in an image where every sweep cell would fail (seen once: missing pyzmq).
verify_ver "sglang-orin:0.4.6-sm87" \
    "sglang-srt-engine" "OK" \
    'python3 -c "import sglang.srt.entrypoints.engine; print(\"sglang-srt-engine OK\")"'
verify_ver "bitsandbytes-bench:r36.4.0" \
    "torch" "2.6.0" \
    'python3 -c "import torch; print(f\"torch {torch.__version__}\")"'
verify_ver "bitsandbytes-bench:r36.4.0" \
    "transformers" "4.57.3" \
    'python3 -c "import transformers; print(f\"transformers {transformers.__version__}\")"'
verify_ver "bitsandbytes-bench:r36.4.0" \
    "bitsandbytes" "0.45.4.dev0" \
    'python3 -c "import bitsandbytes as b; print(f\"bitsandbytes {b.__version__}\")"'

if [ "$VERIFY_FAILS" -gt 0 ]; then
    echo
    echo "  ERROR: ${VERIFY_FAILS} container(s) failed version verify."
    echo "         The sweep would produce a partial CSV that cannot be compared to the paper."
    echo "         Fix the failing container(s) (usually a docker build error above) and re-run."
    exit 3
fi

# Container image sha256 comparison. dustynv/* images are pinned by tag + digest
# in container_digests.txt; if the tag was silently republished with a different
# digest, the sweep may no longer match the paper's numbers. Local-build images
# (sglang-orin, bitsandbytes-bench) get their Id logged but not gated — the
# version-verify above is the meaningful gate for those.
DIGEST_FILE="${HERE}/container_digests.txt"
if [ -f "${DIGEST_FILE}" ]; then
    echo
    echo "  Container digest check vs ${DIGEST_FILE##*/}:"
    while read -r line; do
        [[ "$line" =~ ^#.*$ || -z "${line// }" ]] && continue
        img=$(echo "$line" | awk '{print $1}')
        want=$(echo "$line" | awk '{print $2}')
        got=$(docker image inspect --format '{{.Id}}' "$img" 2>/dev/null)
        if [ "$got" = "$want" ]; then
            echo "    ✓ ${img}"
        elif [[ "$img" == dustynv/* ]]; then
            echo "    WARN ${img} digest drift"
            echo "         want: ${want}"
            echo "         got : ${got}"
        else
            echo "    INFO ${img} local-build Id differs (expected — checked via version-verify)"
        fi
    done < "${DIGEST_FILE}"
else
    echo "  (no ${DIGEST_FILE##*/} — skipping digest comparison)"
fi

# ----------------------------------------------------------------------------
# 3. HF snapshots
# ----------------------------------------------------------------------------
banner "3/5 — HuggingFace snapshots"
# meta-llama/Llama-3.2-1B-Instruct is a GATED model on HuggingFace — the
# HTTP API returns 401 without a valid token even for the .gitattributes
# head request. The earlier (WARN + proceed) behavior caused snapshots to
# populate with just README/LICENSE (128 KB), then F16 GGUF conversion and
# every TRT-LLM engine build failed downstream with confusing error
# messages half an hour into the run. Fail fast with an actionable
# message instead — Bug 24.
if [ -z "${HF_TOKEN:-}" ]; then
    cat <<'EOF' >&2

[FATAL] HF_TOKEN is not set.

  meta-llama/Llama-3.2-1B-Instruct is a *gated* model on HuggingFace and
  cannot be downloaded without an access token. Without it, all TRT-LLM
  engine builds and the local F16 GGUF conversion later in this script
  will fail with cascading errors 20+ minutes into the run.

  To proceed:

    1. Create a HuggingFace read token:
         https://huggingface.co/settings/tokens

    2. Request access to the Llama family (one-time approval, usually
       instant after Meta's licence checkbox):
         https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

    3. Export the token in this shell and re-run:
         export HF_TOKEN=hf_...
         bash prepare_orin.sh

EOF
    exit 4
fi

# `huggingface-cli download --local-dir` with recent huggingface_hub versions
# writes only cache-metadata symlinks under the target; the actual blobs stay
# in ~/.cache/huggingface. Force real file copies so mounts into containers
# work: python -c snapshot_download(local_dir=..., local_dir_use_symlinks=False).
# Revisions pinned to what the paper's shipped sweep was run against.
# Overridable via LLAMA_REV / QWEN_REV env if you need to re-anchor later.
LLAMA_REV="${LLAMA_REV:-9213176726f574b556790deb65791e0c5aa438b6}"
QWEN_REV="${QWEN_REV:-1cfa9a7208912126459214e8b04321603b3df60c}"

py_snapshot() {
    local repo="$1" dest="$2" revision="$3"
    # NOTE: HF_HUB_ENABLE_HF_TRANSFER=1 requires `pip install hf_transfer`
    # which is not universally available. Fall back to the standard HTTP
    # downloader by explicitly unsetting it — slower but works everywhere.
    # `local_dir_use_symlinks` is deprecated in huggingface_hub ≥ 0.23; the
    # newer download path already writes real files under local_dir.
    unset HF_HUB_ENABLE_HF_TRANSFER
    python3 - "$repo" "$dest" "$revision" << 'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
repo, dest, revision = sys.argv[1], sys.argv[2], sys.argv[3]
snapshot_download(
    repo_id=repo,
    local_dir=dest,
    revision=revision,
    token=os.environ.get("HF_TOKEN"),
)
PYEOF
}

LLAMA_SLUG="models--meta-llama--Llama-3.2-1B-Instruct"
LLAMA_DEST="${HF_DIR}/${LLAMA_SLUG}/snapshots/${LLAMA_REV}"
if [ ! -f "${LLAMA_DEST}/config.json" ]; then
    echo "  Downloading meta-llama/Llama-3.2-1B-Instruct @ ${LLAMA_REV:0:12}..."
    py_snapshot meta-llama/Llama-3.2-1B-Instruct "${LLAMA_DEST}" "${LLAMA_REV}"
else
    echo "  cached: ${LLAMA_SLUG}@${LLAMA_REV:0:12}"
fi

QWEN_SLUG="models--Qwen--Qwen3-4B"
QWEN_DEST="${HF_DIR}/${QWEN_SLUG}/snapshots/${QWEN_REV}"
if [ ! -f "${QWEN_DEST}/config.json" ]; then
    echo "  Downloading Qwen/Qwen3-4B @ ${QWEN_REV:0:12}..."
    py_snapshot Qwen/Qwen3-4B "${QWEN_DEST}" "${QWEN_REV}"
else
    echo "  cached: ${QWEN_SLUG}@${QWEN_REV:0:12}"
fi

# ----------------------------------------------------------------------------
# 4. GGUF quants
# ----------------------------------------------------------------------------
banner "4/5 — GGUF quants (Llama-3.2-1B)"
# Q3~Q8 quants (6 files) are sha256-identical to bartowski's published copies
# — verified against our shipped f16/Q4/Q8 fingerprints. Fetched with curl.
# f16 has NO public sha256-identical copy: our f16 (1f33ad43…, 2 479 595 360 B)
# was built locally with llama.cpp `convert_hf_to_gguf.py --outtype f16` off
# the pinned Llama-3.2-1B HF snapshot. We re-run that same conversion here so
# reviewers land on the SAME bytes (and not, e.g., unsloth's F16 which
# differs by 192 bytes / different sha).
BARTOWSKI="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main"

fetch() {
    local url="$1" out="$2"
    if [ -f "$out" ]; then
        echo "  cached: $(basename "$out")"
        return
    fi
    echo "  downloading $(basename "$out") from ${url%/*}"
    if ! curl -L --fail -o "$out" "$url"; then
        rm -f "$out"
        echo "    ERROR: fetch failed for $url"
        return 1
    fi
}

# f16 — LOCAL convert from the pinned HF snapshot (bit-identical to shipped).
# The dustynv/llama_cpp image ships llama.cpp b5283 but its Python venv has
# only numpy — no torch, no transformers, no gguf. Installing torch inside
# aarch64 is slow and requires a Jetson-specific wheel index. Instead we
# extract the convert_hf_to_gguf.py + gguf-py sidecar from the llama.cpp
# image to a host workdir, then run them inside the TensorRT-LLM 0.12
# container (which already ships torch 2.5 + sentencepiece); we only pip
# install `transformers` and `gguf` there — cheap and self-contained.
FP16_GGUF="${GGUF_DIR}/Llama-3.2-1B-Instruct-f16.gguf"
if [ ! -f "${FP16_GGUF}" ]; then
    echo "  Building f16 GGUF locally (llama.cpp b5283 convert + gguf-py from GitHub)..."
    # The dustynv/llama_cpp:b5283 image ships neither the gguf pypi package nor
    # the sibling gguf-py source dir that convert_hf_to_gguf.py expects, and
    # the pypi `gguf` release is skewed against b5283's script (missing
    # CLIP_VISION enum, etc.). Fetch the matching sources directly from the
    # llama.cpp b5283 tag on GitHub (commit 5215b91e9377, published 2025-05-05)
    # into a host workdir, then run under the TensorRT-LLM 0.12 container
    # (which has torch + transformers-compatible Python 3.10).
    LLAMA_CONVERT_WORK="${PROFILE_ROOT}/work/llama_convert"
    if [ ! -f "${LLAMA_CONVERT_WORK}/convert_hf_to_gguf.py" ]; then
        rm -rf "${LLAMA_CONVERT_WORK}"
        mkdir -p "${LLAMA_CONVERT_WORK}"
        echo "    Downloading llama.cpp b5283 tarball..."
        curl -sSL 'https://github.com/ggml-org/llama.cpp/archive/refs/tags/b5283.tar.gz' \
            | tar -xz --strip-components=1 -C "${LLAMA_CONVERT_WORK}" \
                'llama.cpp-b5283/convert_hf_to_gguf.py' \
                'llama.cpp-b5283/gguf-py'
        ls -la "${LLAMA_CONVERT_WORK}" | head -5
    fi
    docker run -i --rm \
        -v "${LLAMA_CONVERT_WORK}:/work:ro" \
        -v "${LLAMA_DEST}:/hf_model:ro" \
        -v "${GGUF_DIR}:/output" \
        dustynv/tensorrt_llm:0.12-r36.4.0 \
        bash -c 'pip install --quiet --no-cache-dir \
                     --index-url https://pypi.org/simple/ \
                     transformers==4.44.2 sentencepiece protobuf && \
                 cd /work && \
                 PYTHONPATH=/work/gguf-py \
                     python3 convert_hf_to_gguf.py /hf_model \
                         --outfile /output/Llama-3.2-1B-Instruct-f16.gguf \
                         --outtype f16'
else
    echo "  cached: $(basename "${FP16_GGUF}")"
fi

# rest — from bartowski (sha256-verified identical to shipped)
for q in Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_L; do
    fetch "${BARTOWSKI}/Llama-3.2-1B-Instruct-${q}.gguf" \
          "${GGUF_DIR}/Llama-3.2-1B-Instruct-${q}.gguf"
done

# ----------------------------------------------------------------------------
# 5. TRT-LLM engines
# ----------------------------------------------------------------------------
banner "5/5 — TRT-LLM engines (Llama-3.2-1B fp16 / int8 / int4)"
LLAMA_SNAP="${LLAMA_DEST}"
if [ ! -f "${LLAMA_SNAP}/config.json" ]; then
    echo "  ERROR: Llama snapshot not found at ${LLAMA_SNAP}"
    exit 1
fi

# TensorRT-LLM 0.12's convert_checkpoint.py crashes on Llama-3.2-1B because
# the model config has tie_word_embeddings=True but stores lm_head as a
# reference to embed_tokens; the postprocess in tensorrt_llm/layers/linear.py
# (line 380) then hits `weights.to(...)` on None. Fix: preprocess the HF
# snapshot into an untied variant — clone embed_tokens into lm_head and flip
# the tie flag — then point convert_checkpoint at the untied dir.
UNTIED_DIR="${HF_DIR}/llama-3.2-1b-instruct-untied"
if [ ! -f "${UNTIED_DIR}/config.json" ]; then
    echo "  Un-tying Llama-3.2-1B lm_head (TRT-LLM 0.12 tied-emb workaround)..."
    mkdir -p "${UNTIED_DIR}"
    # `-i` forwards this shell's stdin (the heredoc) into the container so
    # `python3 -` inside actually receives the script. Without `-i`, docker
    # attaches an empty stdin and python exits with no output.
    docker run -i --rm \
        -v "${LLAMA_SNAP}:/hf_src:ro" \
        -v "${UNTIED_DIR}:/hf_dst" \
        dustynv/tensorrt_llm:0.12-r36.4.0 \
        python3 - << 'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("/hf_src", torch_dtype=torch.float16)
if m.config.tie_word_embeddings:
    with torch.no_grad():
        m.lm_head.weight = torch.nn.Parameter(
            m.model.embed_tokens.weight.detach().clone()
        )
    m.config.tie_word_embeddings = False
    print("  ✓ untied: lm_head cloned from embed_tokens; tie_word_embeddings=False")
else:
    print("  (already untied — nothing to do)")
m.save_pretrained("/hf_dst", safe_serialization=True)
AutoTokenizer.from_pretrained("/hf_src").save_pretrained("/hf_dst")
PYEOF
else
    echo "  cached: ${UNTIED_DIR}"
fi

# Build all 3 quants inside the TRT-LLM container so we get the exact
# TensorRT-LLM 0.12 conversion + build pipeline used by the paper.
TRTLLM_IMG="dustynv/tensorrt_llm:0.12-r36.4.0"
DOCKER_RUN="docker run --rm --runtime nvidia \
    -v ${UNTIED_DIR}:/hf_model \
    -v ${ENGINE_DIR}:/engines \
    -v ${BENCH_DIR}:/benchmarks \
    ${TRTLLM_IMG}"

for quant in fp16 int8 int4; do
    engine_out="${ENGINE_DIR}/llama-3.2-1b-instruct"
    [ "$quant" != "fp16" ] && engine_out="${ENGINE_DIR}/llama-3.2-1b-instruct-${quant}"
    if [ -f "${engine_out}/rank0.engine" ]; then
        echo "  cached: ${engine_out}"
        continue
    fi
    # TRT-LLM fp16 engine build hits transient GPU-memory allocation errors
    # when Docker holds dead/exited containers in state that keeps VRAM
    # bindings pinned. `docker system prune -f` reliably releases them.
    docker system prune -f >/dev/null 2>&1 || true
    echo "  Building TRT-LLM engine: ${quant} → ${engine_out}"
    mkdir -p "${engine_out}"
    case "$quant" in
        fp16)
            $DOCKER_RUN bash -c "cd /opt/TensorRT-LLM/examples/llama && \
                python3 convert_checkpoint.py --model_dir /hf_model --output_dir /engines/ckpt_fp16 --dtype float16 && \
                trtllm-build --checkpoint_dir /engines/ckpt_fp16 --output_dir /engines/llama-3.2-1b-instruct \
                    --gemm_plugin float16 --gpt_attention_plugin float16 \
                    --max_input_len 4096 --max_seq_len 8192 --max_batch_size 1"
            ;;
        int8)
            $DOCKER_RUN bash -c "cd /opt/TensorRT-LLM/examples/llama && \
                python3 convert_checkpoint.py --model_dir /hf_model --output_dir /engines/ckpt_int8 \
                    --dtype float16 --use_weight_only --weight_only_precision int8 && \
                trtllm-build --checkpoint_dir /engines/ckpt_int8 --output_dir /engines/llama-3.2-1b-instruct-int8 \
                    --gemm_plugin float16 --gpt_attention_plugin float16 \
                    --max_input_len 4096 --max_seq_len 8192 --max_batch_size 1"
            ;;
        int4)
            $DOCKER_RUN bash -c "cd /opt/TensorRT-LLM/examples/llama && \
                python3 convert_checkpoint.py --model_dir /hf_model --output_dir /engines/ckpt_int4 \
                    --dtype float16 --use_weight_only --weight_only_precision int4 && \
                trtllm-build --checkpoint_dir /engines/ckpt_int4 --output_dir /engines/llama-3.2-1b-instruct-int4 \
                    --gemm_plugin float16 --gpt_attention_plugin float16 \
                    --max_input_len 4096 --max_seq_len 8192 --max_batch_size 1"
            ;;
    esac
done

# ----------------------------------------------------------------------------
banner "prepare_orin DONE — layout"
echo "  ${HF_DIR}      $(du -sh "${HF_DIR}" 2>/dev/null | awk '{print $1}')"
echo "  ${GGUF_DIR}    $(du -sh "${GGUF_DIR}" 2>/dev/null | awk '{print $1}')"
echo "  ${ENGINE_DIR}  $(du -sh "${ENGINE_DIR}" 2>/dev/null | awk '{print $1}')"
echo
echo "Next: bash run_orin_collection.sh"
