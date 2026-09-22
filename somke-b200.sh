#!/usr/bin/env bash
# B200 wrapper for the upstream-aligned Video-KTR smoke run.
#
# The intentionally retained filename matches the requested operator command.
# It delegates the actual baseline/KTR work to smoke.sh, but owns B200-specific
# capacity checks and the official keep-alive lifecycle.  Nothing in this file
# kills processes directly: the cluster-provided keep-alive controller is the
# only mechanism used to stop or restore that workload.

set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
b200_root="${B200_ROOT:-$(cd "${project_root}/.." && pwd)}"
env_file="${B200_ENV_FILE:-${b200_root}/b200.env}"

# An untracked, machine-local environment file keeps deployment-specific
# controller paths out of Git.  Explicit exported variables take precedence.
if [[ -f "${env_file}" ]]; then
    # shellcheck disable=SC1090
    source "${env_file}"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${B200_RUN_ROOT:-${b200_root}/artifacts/grpo-smoke-b200/${timestamp}}"
python_bin="${PYTHON_BIN:-${b200_root}/venv/bin/python}"
model_path="${B200_MODEL_PATH:-${b200_root}/model/Qwen2.5-VL-7B-COT-SFT}"
data_root="${B200_DATA_ROOT:-${b200_root}/data}"
dataset_source="${B200_DATASET_SOURCE:-${data_root}/Video-R1-Holmes-16k.json}"
keepalive_main="${B200_KEEPALIVE_MAIN:-}"
keepalive_launcher="${B200_KEEPALIVE_LAUNCHER:-}"
allow_pause="${B200_ALLOW_PAUSE_KEEPALIVE:-0}"

# Eight deterministic records verified to be video records.  This makes the
# smoke exercise the visual and temporal paths on every rank, while avoiding
# the mixed-modality distributed sampler path reserved for the full run.
smoke_problem_ids="${B200_SMOKE_PROBLEM_IDS:-9969,10457,10648,10895,11260,11854,11999,12354}"

keepalive_paused=0
smoke_started=0

mkdir -p "${run_root}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] [b200-smoke] %s\n' -1 "$*" | tee -a "${run_root}/terminal.log"
}

process_command_line() {
    local pid="$1"
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline"
}

find_keepalive_pids() {
    local proc pid command_line
    keepalive_pids=()
    for proc in /proc/[0-9]*; do
        [[ -r "${proc}/cmdline" ]] || continue
        pid="${proc##*/}"
        command_line="$(process_command_line "${pid}" 2>/dev/null || true)"
        if [[ "${command_line}" == *"${keepalive_main}"* ]]; then
            keepalive_pids+=("${pid}")
        fi
    done
}

wait_for_keepalive_count() {
    local expected="$1"
    local description="$2"
    local elapsed=0
    while (( elapsed < 60 )); do
        find_keepalive_pids
        if (( ${#keepalive_pids[@]} == expected )); then
            log "keep-alive ${description}: verified ${expected} matching main process(es)"
            return 0
        fi
        sleep 1
        ((elapsed += 1))
    done
    find_keepalive_pids
    log "ERROR: keep-alive ${description}: expected ${expected} matching process(es), observed ${#keepalive_pids[@]} (${keepalive_pids[*]:-none})"
    return 1
}

start_keepalive() {
    find_keepalive_pids
    if (( ${#keepalive_pids[@]} == 1 )); then
        log "keep-alive restore: already running (pid=${keepalive_pids[0]}); no duplicate started"
        keepalive_paused=0
        return 0
    fi
    if (( ${#keepalive_pids[@]} > 1 )); then
        log "ERROR: keep-alive restore is ambiguous (${keepalive_pids[*]}); refusing to start another copy"
        return 1
    fi
    log "keep-alive restore: invoking the official controller"
    KEEP_ALIVE_DASHBOARD=0 bash "${keepalive_launcher}" start 2>&1 | tee -a "${run_root}/keepalive.log"
    wait_for_keepalive_count 1 "restore"
    keepalive_paused=0
}

cleanup() {
    local status=$?
    local restore_status=0
    trap - EXIT
    set +e
    if (( keepalive_paused == 1 )); then
        log "cleanup: smoke ended (exit=${status}); restoring keep-alive through its official controller"
        if ! start_keepalive; then
            restore_status=1
            log "CRITICAL: keep-alive restoration needs operator attention; see ${run_root}/keepalive.log"
        fi
    fi
    if (( status == 0 && restore_status != 0 )); then
        # A completed smoke is not operationally successful if it leaves the
        # shared-GPU protection workload down.  Preserve a preceding training
        # error, but make a restore failure independently visible to callers.
        status=5
    fi
    if (( status == 0 )); then
        log "B200 smoke wrapper completed successfully"
    else
        log "B200 smoke wrapper ended with exit=${status}; artifacts remain at ${run_root}"
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

require_path() {
    local label="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        log "ERROR: ${label} is unavailable: ${path}"
        exit 2
    fi
}

log "phase 1/6: validating B200 host, durable inputs, and local runtime"
if [[ "${allow_pause}" != "1" ]]; then
    log "ERROR: set B200_ALLOW_PAUSE_KEEPALIVE=1 only when you intend this smoke to pause and later restore the known keep-alive"
    exit 2
fi
if [[ -z "${keepalive_main}" || -z "${keepalive_launcher}" ]]; then
    log "ERROR: set B200_KEEPALIVE_MAIN and B200_KEEPALIVE_LAUNCHER in ${env_file} or the environment"
    exit 2
fi
require_path "B200 Python" "${python_bin}"
require_path "model checkpoint" "${model_path}"
require_path "data root" "${data_root}"
require_path "Holmes manifest" "${dataset_source}"
require_path "keep-alive main" "${keepalive_main}"
require_path "keep-alive controller" "${keepalive_launcher}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi is unavailable; run this on the B200 GPU node"
    exit 2
fi
gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d '[:space:]')"
if [[ "${gpu_count}" != "8" ]]; then
    log "ERROR: B200 smoke requires exactly 8 visible physical GPUs; observed ${gpu_count}"
    exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Fv 'B200' >/dev/null; then
    log "ERROR: at least one visible GPU is not a B200; refusing a profile mismatch"
    exit 2
fi

# Do this before the Python gate rather than only before the delegated
# launcher.  A stale scheduler/user mask could otherwise make
# torch.cuda.device_count() report a subset even though this dedicated B200
# node exposes all eight physical cards.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

site_dir="$("${python_bin}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
require_path "venv site-packages" "${site_dir}"

export PYTHONNOUSERSITE=1
# Python consults PYTHONPATH before site-packages.  Put the pinned venv first
# and discard any inherited control-plane paths so phase 2 cannot validate a
# platform Transformers build that differs from the runtime we launch.
export PYTHONPATH="${site_dir}:${project_root}/src:${project_root}/src/r1-v/src/open_r1:${project_root}/src/r1-v/src:${project_root}/src/qwen-vl-utils/src"
export MODEL_PATH="${model_path}"
log "phase 2/6: checking pinned imports, FlashAttention-2 availability, and distinct Qwen media IDs"
"${python_bin}" - <<'PY' 2>&1 | tee -a "${run_root}/terminal.log"
import os
from pathlib import Path

import flash_attn
import torch
import tokenizers
import transformers
import trl
from transformers import AutoConfig

expected = {
    "transformers": "4.49.0.dev0",
    "tokenizers": "0.21.4",
    "trl": "0.16.0",
}
observed = {
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "trl": trl.__version__,
}
if observed != expected:
    raise SystemExit(f"pinned package mismatch: {observed!r} != {expected!r}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 CUDA devices, observed={torch.cuda.device_count()}")
config = AutoConfig.from_pretrained(os.environ["MODEL_PATH"], local_files_only=True)
image_id = int(config.image_token_id)
video_id = int(config.video_token_id)
if image_id == video_id:
    raise SystemExit(f"Qwen image/video token IDs must differ, both were {image_id}")
print(
    "[b200-preflight] "
    f"torch={torch.__version__}; cuda={torch.version.cuda}; "
    f"flash_attn={flash_attn.__version__}; transformers={transformers.__version__}; "
    f"image_token_id={image_id}; video_token_id={video_id}; "
    f"visible={[torch.cuda.get_device_name(i) for i in range(8)]}",
    flush=True,
)
PY

flash_binary="$(find "$(dirname "$("${python_bin}" -c 'import flash_attn; print(flash_attn.__file__)')")" -type f -name '*cuda*.so' -print -quit)"
if [[ -z "${flash_binary}" ]] || ! strings "${flash_binary}" | grep -q 'sm_100'; then
    log "ERROR: installed FlashAttention-2 binary does not advertise sm_100; do not silently fall back to a different attention backend"
    exit 2
fi
log "FlashAttention-2 binary advertises sm_100: ${flash_binary}"

{
    printf 'profile=8xB200-upstream-aligned\n'
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'b200_root=%s\nmodel_path=%s\ndata_root=%s\ndataset_source=%s\n' "${b200_root}" "${model_path}" "${data_root}" "${dataset_source}"
    printf 'holmes_requested=Holmes-16k\nmax_prompt_length=16384\nmax_completion_length=768\nnum_generations=8\nmax_pixels=401408\nnframes=16\n'
    printf 'attn_implementation=flash_attention_2\nselection_scope=per_completion\nselection_ratio=0.2\ndelta=absolute\ntemporal_permutations=1\ntemporal_include_reverse=false\n'
    printf 'image_video_token_ids=distinct\nsmoke_problem_ids=%s\n' "${smoke_problem_ids}"
    printf 'wrapper_sha256=%s\n' "$(sha256sum "${project_root}/somke-b200.sh" | awk '{print $1}')"
    printf 'delegate_sha256=%s\n' "$(sha256sum "${project_root}/smoke.sh" | awk '{print $1}')"
} > "${run_root}/b200_profile.txt"

log "phase 3/6: validating the known keep-alive before pausing it"
find_keepalive_pids
if (( ${#keepalive_pids[@]} == 0 )); then
    log "keep-alive is not running; starting it through the official controller before lifecycle validation"
    start_keepalive
elif (( ${#keepalive_pids[@]} != 1 )); then
    log "ERROR: expected exactly one known keep-alive main process, observed ${#keepalive_pids[@]} (${keepalive_pids[*]})"
    exit 2
else
    log "keep-alive preflight: known main process is pid=${keepalive_pids[0]}"
fi

log "phase 4/6: stopping keep-alive through its official controller; it remains paused only while smoke work owns the GPUs"
keepalive_paused=1
KEEP_ALIVE_DASHBOARD=0 bash "${keepalive_launcher}" stop 2>&1 | tee -a "${run_root}/keepalive.log"
wait_for_keepalive_count 0 "pause"

log "phase 5/6: launching real-time 8xB200 baseline and Video-KTR optimizer-step smoke"
export PYTHON_BIN="${python_bin}"
export GRPO_SITE_PACKAGES="${site_dir}"
export MODEL_PATH="${model_path}"
export DATA_ROOT="${data_root}"
export DATASET_SOURCE="${dataset_source}"
export OUTPUT_ROOT="${b200_root}/artifacts/grpo-smoke-b200"
export RUN_ROOT="${run_root}"
export NPROC_PER_NODE=8
export REQUIRED_NPROC_PER_NODE=8
export EXPECTED_GPU_NAME=B200
export SMOKE_PROFILE_NAME=8xB200-upstream-aligned
export MIN_FREE_GIB=170
export MAX_PROMPT_LENGTH=16384
export MAX_COMPLETION_LENGTH=768
export NUM_GENERATIONS=8
export MAX_PIXELS=401408
export NFRAMES=16
export TEMPORAL_PERMUTATIONS=1
export TEMPORAL_INCLUDE_REVERSE=false
export SMOKE_DATA_TYPE=video
export SMOKE_PROBLEM_IDS="${smoke_problem_ids}"
export ATTN_IMPLEMENTATION=flash_attention_2
export VARIANT="${VARIANT:-both}"
export VIDEO_KTR_RANK_TRACE=1
export VIDEO_KTR_REQUIRE_ATTN_IMPLEMENTATION=flash_attention_2
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-COLL}"
smoke_started=1
"${project_root}/smoke.sh"
smoke_started=0

log "phase 6/6: smoke passed; the EXIT handler will now restore keep-alive and verify it"
