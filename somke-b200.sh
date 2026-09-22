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
explicit_b200_names=()
explicit_b200_values=()
for name in \
    B200_ROOT B200_ENV_FILE B200_RUN_ROOT B200_MODEL_PATH B200_DATA_ROOT \
    B200_DATASET_SOURCE B200_KEEPALIVE_MAIN B200_KEEPALIVE_LAUNCHER \
    B200_ALLOW_PAUSE_KEEPALIVE B200_SMOKE_PROBLEM_IDS; do
    if [[ -v "${name}" ]]; then
        explicit_b200_names+=("${name}")
        explicit_b200_values+=("${!name}")
    fi
done
if [[ -f "${env_file}" ]]; then
    # shellcheck disable=SC1090
    source "${env_file}"
fi
for index in "${!explicit_b200_names[@]}"; do
    export "${explicit_b200_names[index]}=${explicit_b200_values[index]}"
done
# The local env file may provide B200_ROOT when the operator did not export
# one; derive all subsequent default paths only after that precedence merge.
b200_root="${B200_ROOT:-$(cd "${project_root}/.." && pwd)}"

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
# A controller stop command may return an error after it has already stopped
# its workload.  Keep this distinct from a confirmed pause so the EXIT trap
# can inspect the exact process state and restore safely in either outcome.
keepalive_stop_requested=0
delegated_pid=""
delegated_pgid=""
keepalive_original_pid=""
keepalive_original_ticks=""

mkdir -p "${run_root}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] [b200-smoke] %s\n' -1 "$*" | tee -a "${run_root}/terminal.log"
}

process_start_ticks() {
    local pid="$1"
    [[ -r "/proc/${pid}/stat" ]] || return 1
    awk '{print $22}' "/proc/${pid}/stat"
}

process_has_exact_keepalive_main() {
    local pid="$1"
    local argument
    local -a arguments=()
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    mapfile -d '' -t arguments < "/proc/${pid}/cmdline"
    for argument in "${arguments[@]}"; do
        if [[ "${argument}" == "${keepalive_main}" ]]; then
            return 0
        fi
        if [[ -e "${argument}" && "$(realpath -e "${argument}" 2>/dev/null || true)" == "${keepalive_main}" ]]; then
            return 0
        fi
    done
    return 1
}

same_keepalive_identity() {
    local pid="$1"
    local ticks="$2"
    [[ "$(process_start_ticks "${pid}" 2>/dev/null || true)" == "${ticks}" ]] \
        && process_has_exact_keepalive_main "${pid}"
}

capture_keepalive_identity() {
    find_keepalive_pids
    if (( ${#keepalive_pids[@]} != 1 )); then
        log "ERROR: cannot capture keep-alive identity; observed ${#keepalive_pids[@]} matching main process(es)"
        return 1
    fi
    keepalive_original_pid="${keepalive_pids[0]}"
    keepalive_original_ticks="$(process_start_ticks "${keepalive_original_pid}" 2>/dev/null || true)"
    if [[ -z "${keepalive_original_ticks}" ]] \
        || ! same_keepalive_identity "${keepalive_original_pid}" "${keepalive_original_ticks}"; then
        log "ERROR: keep-alive PID identity changed during validation"
        return 1
    fi
    log "keep-alive identity: validated pid=${keepalive_original_pid} start_ticks=${keepalive_original_ticks}"
}

find_keepalive_pids() {
    local proc pid
    keepalive_pids=()
    for proc in /proc/[0-9]*; do
        [[ -r "${proc}/cmdline" ]] || continue
        pid="${proc##*/}"
        if process_has_exact_keepalive_main "${pid}"; then
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

process_is_live() {
    local pid="${1:-}"
    local state
    [[ -n "${pid}" && -r "/proc/${pid}/stat" ]] || return 1
    state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
    [[ -n "${state}" && "${state}" != "Z" ]]
}

query_gpu_compute_pids() {
    local snapshot
    if ! snapshot="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
        return 1
    fi
    mapfile -t gpu_compute_pid_snapshot < <(
        printf '%s\n' "${snapshot}" \
            | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ { gsub(/[[:space:]]/, ""); print }'
    )
}

wait_for_gpu_quiescence() {
    local description="$1"
    local elapsed=0
    gpu_compute_pid_snapshot=()
    while (( elapsed < 60 )); do
        if ! query_gpu_compute_pids; then
            log "CRITICAL: GPU quiescence ${description}: nvidia-smi compute-process query failed"
            return 1
        fi
        if (( ${#gpu_compute_pid_snapshot[@]} == 0 )); then
            log "GPU quiescence ${description}: verified no compute processes"
            return 0
        fi
        sleep 1
        ((elapsed += 1))
    done
    if ! query_gpu_compute_pids; then
        log "CRITICAL: GPU quiescence ${description}: nvidia-smi compute-process query failed"
        return 1
    fi
    log "CRITICAL: GPU quiescence ${description}: compute process(es) remain (${gpu_compute_pid_snapshot[*]:-unknown}); refusing a keep-alive collision"
    return 1
}

stop_delegated_smoke() {
    local pid="${delegated_pid:-}"
    local pgid="${delegated_pgid:-}"
    local elapsed=0
    [[ -n "${pid}" ]] || return 0
    if process_is_live "${pid}"; then
        if [[ -n "${pgid}" && "${pgid}" == "${pid}" ]]; then
            log "cleanup: forwarding TERM to delegated smoke process group=${pgid} before any keep-alive restoration"
            kill -TERM -- "-${pgid}" 2>/dev/null || true
        else
            log "cleanup: forwarding TERM to delegated smoke pid=${pid} before any keep-alive restoration"
            kill -TERM "${pid}" 2>/dev/null || true
        fi
        while process_is_live "${pid}"; do
            if (( elapsed >= 90 )); then
                log "CRITICAL: delegated smoke pid=${pid} did not stop within 90s; keep-alive remains paused"
                return 1
            fi
            sleep 1
            ((elapsed += 1))
        done
    fi
    wait "${pid}" 2>/dev/null || true
    delegated_pid=""
    delegated_pgid=""
    return 0
}

start_keepalive() {
    find_keepalive_pids
    if (( ${#keepalive_pids[@]} == 1 )); then
        log "keep-alive restore: already running (pid=${keepalive_pids[0]}); no duplicate started"
        keepalive_paused=0
        keepalive_stop_requested=0
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
    keepalive_stop_requested=0
}

cleanup() {
    local status=$?
    local restore_status=0
    local child_stopped=1
    trap - EXIT
    # The first signal brought us here.  Do not let a second TERM/INT/HUP
    # interrupt the stop-child → GPU-quiescence → official-restore sequence.
    trap '' INT TERM HUP
    set +e
    if ! stop_delegated_smoke; then
        child_stopped=0
    fi
    # Do not equate a controller stop request with a completed pause.  The
    # controller can fail after stopping its child, or fail before changing
    # anything.  Reconcile the exact keep-alive main process before deciding
    # whether restoration is necessary, so cleanup never starts a duplicate.
    if (( keepalive_paused == 0 && keepalive_stop_requested == 1 )); then
        find_keepalive_pids
        if (( ${#keepalive_pids[@]} == 0 )); then
            keepalive_paused=1
            log "cleanup: controller stop request left no verified keep-alive process; restoration is required"
        elif (( ${#keepalive_pids[@]} == 1 )); then
            keepalive_stop_requested=0
            log "cleanup: keep-alive remained active after failed stop request; no duplicate restore is needed"
        else
            restore_status=1
            log "CRITICAL: keep-alive state is ambiguous after failed stop request (${keepalive_pids[*]})"
        fi
    fi
    if (( keepalive_paused == 1 )); then
        if (( child_stopped != 1 )); then
            restore_status=1
            log "CRITICAL: keep-alive remains paused because delegated smoke may still own GPUs"
        elif ! wait_for_gpu_quiescence "before keep-alive restore"; then
            restore_status=1
            log "CRITICAL: keep-alive remains paused because a compute process still owns a GPU"
        else
            log "cleanup: smoke ended (exit=${status}); restoring keep-alive through its official controller"
            if ! start_keepalive; then
                restore_status=1
                log "CRITICAL: keep-alive restoration needs operator attention; see ${run_root}/keepalive.log"
            fi
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
trap 'exit 129' HUP

require_path() {
    local label="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        log "ERROR: ${label} is unavailable: ${path}"
        exit 2
    fi
}

log "phase 1/7: validating B200 host, durable inputs, and local runtime"
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
keepalive_main="$(realpath -e "${keepalive_main}")"
keepalive_launcher="$(realpath -e "${keepalive_launcher}")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi is unavailable; run this on the B200 GPU node"
    exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
    log "ERROR: setsid is unavailable; cannot isolate the delegated smoke process group for safe cancellation"
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
export SETUPTOOLS_USE_DISTUTILS=local
export FORCE_QWENVL_VIDEO_READER=torchvision
# Python consults PYTHONPATH before site-packages.  Put the pinned venv first
# and discard any inherited control-plane paths so phase 2 cannot validate a
# platform Transformers build that differs from the runtime we launch.
export PYTHONPATH="${site_dir}:${project_root}/src:${project_root}/src/r1-v/src/open_r1:${project_root}/src/r1-v/src:${project_root}/src/qwen-vl-utils/src"
export MODEL_PATH="${model_path}"

# Triton's NVIDIA driver helper is compiled lazily on its first rotary-kernel
# invocation.  The B200 image has CUDA headers, but they are not on GCC's
# implicit include path in this isolated runtime.  Resolve an explicit,
# overrideable include directory and prove it with the same compiler Triton
# will use before pausing the external keep-alive workload.
triton_cuda_include_dir="${TRITON_CUDA_INCLUDE_DIR:-}"
if [[ -z "${triton_cuda_include_dir}" ]]; then
    for candidate in "${CUDA_HOME:+${CUDA_HOME}/include}" /usr/local/cuda/include; do
        if [[ -n "${candidate}" && -f "${candidate}/cuda.h" ]]; then
            triton_cuda_include_dir="${candidate}"
            break
        fi
    done
fi
if [[ -z "${triton_cuda_include_dir}" || ! -f "${triton_cuda_include_dir}/cuda.h" ]]; then
    log "ERROR: Triton needs a CUDA include directory containing cuda.h; set TRITON_CUDA_INCLUDE_DIR"
    exit 2
fi
triton_cuda_include_dir="$(realpath -e "${triton_cuda_include_dir}")"
if ! command -v gcc >/dev/null 2>&1; then
    log "ERROR: gcc is unavailable; Triton cannot build its CUDA driver helper"
    exit 2
fi
if ! printf '#include <cuda.h>\n' | gcc -I"${triton_cuda_include_dir}" -x c -E -o /dev/null -; then
    log "ERROR: gcc cannot preprocess cuda.h from ${triton_cuda_include_dir}"
    exit 2
fi
export C_INCLUDE_PATH="${triton_cuda_include_dir}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="${triton_cuda_include_dir}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
triton_ptxas_path="${TRITON_PTXAS_PATH:-}"
if [[ -z "${triton_ptxas_path}" ]]; then
    for candidate in "${CUDA_HOME:+${CUDA_HOME}/bin/ptxas}" /usr/local/cuda/bin/ptxas; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            triton_ptxas_path="${candidate}"
            break
        fi
    done
fi
if [[ -z "${triton_ptxas_path}" || ! -x "${triton_ptxas_path}" ]]; then
    log "ERROR: Triton needs an executable ptxas; set TRITON_PTXAS_PATH"
    exit 2
fi
triton_ptxas_path="$(realpath -e "${triton_ptxas_path}")"
if ! "${triton_ptxas_path}" --version | grep -Eq 'release [0-9]+\.[0-9]+'; then
    log "ERROR: selected ptxas does not report a usable CUDA release: ${triton_ptxas_path}"
    exit 2
fi
# Triton 3.6 only searches its bundled tool directory unless this knob is set;
# it deliberately does not fall back to PATH.  Keep the selected image's
# matching ptxas explicit and record it in the run profile.
export TRITON_PTXAS_PATH="${triton_ptxas_path}"
triton_tmpdir="${B200_TRITON_TMPDIR:-${b200_root}/.triton-tmp}"
triton_cache_dir="${B200_TRITON_CACHE_DIR:-${b200_root}/.triton-cache}"
for triton_path in "${triton_tmpdir}" "${triton_cache_dir}"; do
    [[ "${triton_path}" == /* ]] || { log "ERROR: Triton path must be absolute: ${triton_path}"; exit 2; }
    case "${triton_path}" in
        /tmp|/tmp/*|/mnt|/mnt/*)
            log "ERROR: Triton cache/temp path must not use /tmp or /mnt: ${triton_path}"
            exit 2
            ;;
    esac
    mkdir -p "${triton_path}"
done
export TMPDIR="${triton_tmpdir}"
export TRITON_CACHE_DIR="${triton_cache_dir}"
log "Triton JIT preflight: cuda_headers=${triton_cuda_include_dir}; ptxas=${triton_ptxas_path}; cache=${triton_cache_dir}; temp=${triton_tmpdir}"
log "phase 2/7: checking pinned imports, FlashAttention-2 availability, and distinct Qwen media IDs"
"${python_bin}" - <<'PY' 2>&1 | tee -a "${run_root}/terminal.log"
import os
from importlib import metadata
from pathlib import Path

import flash_attn
import torch
import tokenizers
import transformers
import triton
import triton_kernels
import trl
from transformers import AutoConfig

expected = {
    "transformers": "4.49.0.dev0",
    "tokenizers": "0.21.4",
    "trl": "0.16.0",
    "triton": "3.6.0",
    "pytorch_triton": "3.6.0+git9844da95.nv26.2",
    "triton_kernels": "1.0.0+git9844da95.nv26.2",
}
observed = {
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "trl": trl.__version__,
    "triton": triton.__version__,
    "pytorch_triton": metadata.version("pytorch-triton"),
    "triton_kernels": metadata.version("triton_kernels"),
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

# The FA2 CUDA extension is commonly a top-level module rather than a file
# beneath the Python ``flash_attn`` package.  Resolve it through the module
# loader so an isolated runtime bridge cannot make this gate miss a valid FA2
# binary (or pass against a different package directory).
# FA2's extension links against libtorch.  Import torch first so its shared
# libraries are loaded before resolving the top-level FA2 extension in this
# short standalone subprocess; the main Python preflight already does this.
flash_binary="$("${python_bin}" -c 'import torch; import flash_attn_2_cuda; print(flash_attn_2_cuda.__file__)')"
if [[ -z "${flash_binary}" ]] || ! "${python_bin}" - "${flash_binary}" <<'PY'
from pathlib import Path
import sys

payload = Path(sys.argv[1]).read_bytes().lower()
raise SystemExit(0 if b"sm_100" in payload or b"sm100" in payload or b"compute_100" in payload else 1)
PY
then
    log "ERROR: installed FlashAttention-2 binary does not contain an sm_100 marker; do not silently fall back to a different attention backend"
    exit 2
fi
log "FlashAttention-2 binary contains an sm_100 marker: ${flash_binary}"

{
    printf 'profile=8xB200-upstream-aligned\n'
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'b200_root=%s\nmodel_path=%s\ndata_root=%s\ndataset_source=%s\n' "${b200_root}" "${model_path}" "${data_root}" "${dataset_source}"
    # The upstream launcher leaves nframes unset; GRPOScriptArguments defaults
    # it to 8.  Keep that value explicit in this reproduction profile rather
    # than retaining the earlier 16-frame stress setting.
    printf 'holmes_requested=Holmes-16k\nmax_prompt_length=16384\nmax_completion_length=768\nnum_generations=8\nmax_pixels=401408\nnframes=8\n'
    printf 'attn_implementation=flash_attention_2\nqwen_fa2_rotary_dtype_compat=true\nselection_scope=per_completion\nselection_ratio=0.2\ndelta=absolute\ntemporal_permutations=1\ntemporal_include_reverse=false\n'
    printf 'triton_cuda_include_dir=%s\ntriton_ptxas_path=%s\ntriton_cache_dir=%s\ntriton_tmpdir=%s\n' "${triton_cuda_include_dir}" "${triton_ptxas_path}" "${triton_cache_dir}" "${triton_tmpdir}"
    printf 'image_video_token_ids=distinct\nvideo_reader_backend=torchvision\nsmoke_problem_ids=%s\n' "${smoke_problem_ids}"
    printf 'wrapper_sha256=%s\n' "$(sha256sum "${project_root}/somke-b200.sh" | awk '{print $1}')"
    printf 'delegate_sha256=%s\n' "$(sha256sum "${project_root}/smoke.sh" | awk '{print $1}')"
} > "${run_root}/b200_profile.txt"

log "phase 3/7: validating the known keep-alive before pausing it"
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
capture_keepalive_identity

log "phase 4/7: stopping keep-alive through its official controller; it remains paused only while smoke work owns the GPUs"
if ! same_keepalive_identity "${keepalive_original_pid}" "${keepalive_original_ticks}"; then
    log "ERROR: keep-alive PID identity changed before official stop; refusing to manage it"
    exit 2
fi
keepalive_stop_requested=1
KEEP_ALIVE_DASHBOARD=0 bash "${keepalive_launcher}" stop 2>&1 | tee -a "${run_root}/keepalive.log"
wait_for_keepalive_count 0 "pause"
keepalive_paused=1
wait_for_gpu_quiescence "after keep-alive pause"

log "phase 5/7: executing tiny FA2 + the real Qwen rotary forward/backward on B200 before distributed training"
PYTHONPATH="${project_root}/src/r1-v/src${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - <<'PY' 2>&1 | tee -a "${run_root}/terminal.log"
import torch
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import apply_rotary
from open_r1.trainer.qwen25vl_fa2_compat import install_qwen25vl_fa2_rotary_dtype_compat
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

device = "cuda:0"
q = torch.randn((1, 16, 2, 64), device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
attention = flash_attn_func(q, k, v, causal=False)
cos = torch.randn((16, 32), device=device, dtype=torch.bfloat16)
sin = torch.randn_like(cos)
rotary = apply_rotary(q, cos, sin, interleaved=False, inplace=False)
assert install_qwen25vl_fa2_rotary_dtype_compat(modeling_qwen2_5_vl)
qwen_cos = torch.randn((16, 64), device=device, dtype=torch.bfloat16)
qwen_sin = torch.randn_like(qwen_cos)
qwen_q, qwen_k = modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt(q, k, qwen_cos, qwen_sin)
assert qwen_q.dtype == torch.bfloat16 and qwen_k.dtype == torch.bfloat16
(attention.float().square().mean() + rotary.float().square().mean() + qwen_q.float().square().mean() + qwen_k.float().square().mean()).backward()
torch.cuda.synchronize(0)
print("[b200-preflight] fa2_and_qwen_fp32_rope_rotary_forward_backward=ok", flush=True)
PY

log "phase 6/7: launching real-time 8xB200 baseline and Video-KTR optimizer-step smoke"
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
export NFRAMES=8
export TEMPORAL_PERMUTATIONS=1
export TEMPORAL_INCLUDE_REVERSE=false
export SMOKE_DATA_TYPE=video
export SMOKE_PROBLEM_IDS="${smoke_problem_ids}"
export ATTN_IMPLEMENTATION=flash_attention_2
export VARIANT="${VARIANT:-both}"
export VIDEO_KTR_RANK_TRACE=1
export VIDEO_KTR_REQUIRE_ATTN_IMPLEMENTATION=flash_attention_2
export VIDEO_KTR_QWEN_FA2_ROTARY_DTYPE_COMPAT=1
# PyTorch's B200 image renamed this setting.  Drop any inherited deprecated
# spelling so distributed workers do not emit a warning on every rank.
unset TORCH_NCCL_TRACE_BUFFER_SIZE
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-COLL}"
set +e
setsid "${project_root}/smoke.sh" &
delegated_pid="$!"
for ((isolation_attempt = 0; isolation_attempt < 30; isolation_attempt += 1)); do
    delegated_pgid="$(ps -o pgid= -p "${delegated_pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "${delegated_pgid}" == "${delegated_pid}" ]] || ! process_is_live "${delegated_pid}"; then
        break
    fi
    sleep 0.1
done
if process_is_live "${delegated_pid}" && [[ "${delegated_pgid}" != "${delegated_pid}" ]]; then
    log "ERROR: delegated smoke did not create an isolated process group (pid=${delegated_pid}, pgid=${delegated_pgid:-unknown})"
    exit 2
fi
wait "${delegated_pid}"
delegated_status="$?"
delegated_pid=""
delegated_pgid=""
set -e
if (( delegated_status != 0 )); then
    log "delegated smoke failed with exit=${delegated_status}"
    exit "${delegated_status}"
fi

log "phase 7/7: smoke passed; the EXIT handler will now restore keep-alive and verify it"
