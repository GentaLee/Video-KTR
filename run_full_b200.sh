#!/usr/bin/env bash
# Manual, upstream-aligned full GRPO launcher for the isolated 8xB200 profile.
#
# This file is intentionally separate from the historical H200 launcher.  It
# owns the B200 keep-alive lifecycle and preserves the Holmes image+video
# mixture, while the H200 launcher remains a historical diagnostic path.  The
# full job is never started by setup or smoke scripts; an operator must invoke
# this file explicitly after reading its data-class gate.

set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
b200_root="${B200_ROOT:-$(cd "${project_root}/.." && pwd)}"
b200_root="$(realpath -m "${b200_root}")"
env_file="${B200_ENV_FILE:-${b200_root}/b200.env}"

# Machine-local controller identities live outside Git.  Preserve explicitly
# exported values across source so an operator can safely override one entry
# for a single invocation without editing the private environment file.
explicit_b200_names=()
explicit_b200_values=()
for name in \
    B200_ROOT B200_ENV_FILE B200_RUN_ROOT RUN_ROOT VARIANT MAX_STEPS PYTHON_BIN B200_MODEL_PATH B200_DATA_ROOT B200_RESUME_FROM_CHECKPOINT \
    B200_DATASET_SOURCE B200_KEEPALIVE_MAIN B200_KEEPALIVE_LAUNCHER \
    B200_ALLOW_PAUSE_KEEPALIVE B200_ALLOW_REDUCED_HOLMES \
    B200_ALLOW_DECODER_FILTERED B200_SMOKE_RUN_ROOT B200_ENV_MANIFEST \
    B200_MODEL_INTEGRITY_MANIFEST B200_MODEL_SHA256_MANIFEST \
    B200_MEDIA_CACHE_DIR B200_MEDIA_DECODE_THREADS B200_DATALOADER_WORKERS \
    B200_MEDIA_DECODE_WORKERS B200_MEDIA_DECODE_TIMEOUT_SECONDS \
    B200_MEDIA_DECODE_PROGRESS_EVERY B200_MEDIA_DECODE_STATUS_SECONDS; do
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
b200_root="${B200_ROOT:-$(cd "${project_root}/.." && pwd)}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
variant="${VARIANT:-ktr}"
run_root="${B200_RUN_ROOT:-${RUN_ROOT:-${b200_root}/artifacts/grpo-full-b200/${variant}-${timestamp}}}"
run_root="$(realpath -m "${run_root}")"
python_bin="${PYTHON_BIN:-${b200_root}/venv/bin/python}"
model_path="${B200_MODEL_PATH:-${b200_root}/model/Qwen2.5-VL-7B-COT-SFT}"
data_root="${B200_DATA_ROOT:-${b200_root}/data}"
dataset_source="${B200_DATASET_SOURCE:-${data_root}/Video-R1-Holmes-16k.json}"
smoke_root="${B200_SMOKE_RUN_ROOT:-${b200_root}/artifacts/grpo-smoke-b200/latest-passed}"
environment_manifest="${B200_ENV_MANIFEST:-${b200_root}/environment-manifest.json}"
model_integrity_manifest="${B200_MODEL_INTEGRITY_MANIFEST:-${B200_MODEL_SHA256_MANIFEST:-${project_root}/manifests/video-r1-qwen25vl-7b-cot-sft-f71f0f1e22c015007fccd080eef87824fe292a10.json}}"
keepalive_main="${B200_KEEPALIVE_MAIN:-}"
keepalive_launcher="${B200_KEEPALIVE_LAUNCHER:-}"
allow_pause="${B200_ALLOW_PAUSE_KEEPALIVE:-0}"
allow_reduced="${B200_ALLOW_REDUCED_HOLMES:-0}"
allow_decoder_filtered="${B200_ALLOW_DECODER_FILTERED:-0}"

# The training shape is fixed by the upstream-aligned B200 profile.  MAX_STEPS
# is overrideable for bounded diagnostics; media decode controls affect only
# the CPU preflight, not the dataset or training shape.
nproc_per_node=8
max_prompt_length=16384
max_completion_length=768
num_generations=8
max_pixels=401408
nframes=8
temporal_permutations=1
temporal_include_reverse=false
max_steps="${MAX_STEPS:--1}"
save_steps=100
save_total_limit=2
heartbeat_seconds="${HEARTBEAT_SECONDS:-30}"
gpu_sample_seconds="${GPU_SAMPLE_SECONDS:-5}"
decoder_workers="${B200_MEDIA_DECODE_WORKERS:-8}"
decoder_timeout_seconds="${B200_MEDIA_DECODE_TIMEOUT_SECONDS:-600}"
decoder_progress_every="${B200_MEDIA_DECODE_PROGRESS_EVERY:-100}"
decoder_status_seconds="${B200_MEDIA_DECODE_STATUS_SECONDS:-30}"
decoder_threads="${B200_MEDIA_DECODE_THREADS:-1}"
media_cache_dir="${B200_MEDIA_CACHE_DIR:-${b200_root}/cache/qwen-media-v1}"
dataloader_workers="${B200_DATALOADER_WORKERS:-2}"
ddp_timeout_seconds="${DDP_TIMEOUT_SECONDS:-600}"
min_free_gib=170
full_output_selected_token="${FULL_OUTPUT_SELECTED_TOKEN:-false}"
token_record_limit="${TOKEN_RECORD_LIMIT:-0}"

path_dataset="${run_root}/Holmes-16k-path-verified.json"
path_manifest="${path_dataset}.manifest.json"
decoder_dataset="${run_root}/Holmes-16k-media-decoder-verified.json"
decoder_manifest="${decoder_dataset}.manifest.json"
smoke_gate="${run_root}/smoke_gate.json"
smoke_gate_pre_gpu="${run_root}/smoke_gate_pre_gpu.json"
runtime_gate="${run_root}/runtime_gate.json"
runtime_gate_pre_gpu="${run_root}/runtime_gate_pre_gpu.json"
path_gate="${run_root}/data_path_gate.json"
data_gate="${run_root}/data_gate.json"

keepalive_paused=0
keepalive_managed=0
keepalive_stop_requested=0
keepalive_original_pid=""
keepalive_original_ticks=""
preflight_pid=""
preflight_label=""
training_pid=""
monitor_pid=""
heartbeat_pid=""
log_follower_pid=""
training_started=""
training_ended=""
training_exit=""
summary_written=0

is_ephemeral_path() {
    case "$1" in
        /tmp|/tmp/*|/mnt|/mnt/*) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ "${b200_root}" != /* || "${run_root}" != /* ]] || is_ephemeral_path "${b200_root}" || is_ephemeral_path "${run_root}"; then
    printf '[b200-full] ERROR: B200_ROOT and RUN_ROOT must use durable storage, not /tmp or /mnt\n' >&2
    exit 2
fi
if [[ "${environment_manifest}" != /* || "${model_integrity_manifest}" != /* ]]; then
    printf '[b200-full] ERROR: B200 environment/model integrity manifests must be absolute paths\n' >&2
    exit 2
fi
if [[ "${media_cache_dir}" != /* ]] || is_ephemeral_path "${media_cache_dir}"; then
    printf '[b200-full] ERROR: media cache must use an absolute durable path\n' >&2
    exit 2
fi
# Share this lock with smoke: two operators must not independently stop/start
# protection or launch KTR and baseline on the same eight GPUs.
exec 9>"${b200_root}/.b200-training.lock"
if ! flock -n 9; then
    printf '[b200-full] ERROR: another B200 smoke/full task owns this node; run variants serially\n' >&2
    exit 2
fi
if [[ -e "${run_root}" ]]; then
    printf '[b200-full] ERROR: RUN_ROOT must name a new directory: %s\n' "${run_root}" >&2
    exit 2
fi
if ! mkdir -p "$(dirname "${run_root}")" || ! mkdir "${run_root}"; then
    printf '[b200-full] ERROR: cannot atomically claim RUN_ROOT: %s\n' "${run_root}" >&2
    exit 2
fi

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] [b200-full] %s\n' -1 "$*" >> "${run_root}/terminal.log"
}

# Producers write regular files. A slow, disconnected or flow-controlled IDE
# terminal may block this observer, but can never block decoder supervision or
# torchrun. The detached entrypoint also survives terminal disconnects.
touch "${run_root}/terminal.log"
tail --pid="$$" --sleep-interval=0.2 -n +1 -f "${run_root}/terminal.log" &
log_follower_pid="$!"

require_path() {
    local label="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        log "ERROR: ${label} is unavailable: ${path}"
        exit 2
    fi
}

require_positive_integer() {
    local label="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        log "ERROR: ${label} must be a positive integer; got '${value}'"
        exit 2
    fi
}

normalize_bool() {
    local label="$1"
    local value="$2"
    case "${value}" in
        1|true|TRUE|yes|YES) printf 'true\n' ;;
        0|false|FALSE|no|NO) printf 'false\n' ;;
        *)
            log "ERROR: ${label} must be 0/1 or false/true; got '${value}'"
            return 2
            ;;
    esac
}

process_is_live() {
    local pid="${1:-}"
    local state
    [[ -n "${pid}" && -r "/proc/${pid}/stat" ]] || return 1
    state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
    [[ -n "${state}" && "${state}" != "Z" ]]
}

process_group_is_live() {
    local process_group_id="${1:-}"
    [[ "${process_group_id}" =~ ^[1-9][0-9]*$ ]] || return 1
    ps -eo pgid=,stat= | awk -v group="${process_group_id}" \
        '$1 == group && $2 !~ /^Z/ { found=1 } END { exit !found }'
}

stop_child() {
    local child_pid="${1:-}"
    if process_is_live "${child_pid}"; then
        kill -TERM "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
}

stop_owned_process_group() {
    local label="$1"
    local pid="$2"
    local elapsed=0
    [[ -n "${pid}" ]] || return 0
    if process_group_is_live "${pid}"; then
        log "cleanup: forwarding TERM to ${label} process group=${pid} before any keep-alive restoration"
        kill -TERM -- "-${pid}" 2>/dev/null || true
        while process_group_is_live "${pid}"; do
            if (( elapsed >= 90 )); then
                log "CRITICAL: ${label} process group=${pid} did not stop within 90s; keep-alive remains paused"
                return 1
            fi
            sleep 1
            ((elapsed += 1))
        done
    fi
    wait "${pid}" 2>/dev/null || true
    return 0
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

same_keepalive_identity() {
    local pid="$1"
    local ticks="$2"
    [[ "$(process_start_ticks "${pid}" 2>/dev/null || true)" == "${ticks}" ]] \
        && process_has_exact_keepalive_main "${pid}"
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
    log "ERROR: keep-alive ${description}: expected ${expected}, observed ${#keepalive_pids[@]} (${keepalive_pids[*]:-none})"
    return 1
}

query_gpu_compute_pids() {
    local snapshot
    if ! snapshot="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
        return 1
    fi
    mapfile -t gpu_compute_pid_snapshot < <(
        printf '%s\n' "${snapshot}" | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ { gsub(/[[:space:]]/, ""); print }'
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
    query_gpu_compute_pids || true
    log "CRITICAL: GPU quiescence ${description}: compute process(es) remain (${gpu_compute_pid_snapshot[*]:-unknown})"
    return 1
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
    log "keep-alive identity validated before managed pause"
}

start_keepalive() {
    find_keepalive_pids
    if (( ${#keepalive_pids[@]} == 1 )); then
        log "keep-alive restore: already running; no duplicate started"
        keepalive_paused=0
        keepalive_stop_requested=0
        return 0
    fi
    if (( ${#keepalive_pids[@]} > 1 )); then
        log "ERROR: keep-alive restore is ambiguous (${keepalive_pids[*]}); refusing another copy"
        return 1
    fi
    wait_for_gpu_quiescence "before starting an absent keep-alive" || return 1
    log "keep-alive restore: invoking official controller"
    KEEP_ALIVE_DASHBOARD=0 bash "${keepalive_launcher}" start 9>&- 2>&1 | tee -a "${run_root}/keepalive.log" >> "${run_root}/terminal.log"
    wait_for_keepalive_count 1 "restore"
    keepalive_paused=0
    keepalive_stop_requested=0
}

ensure_cpu_keepalive() {
    # Called only by the controlling shell while CPU preflight owns the node.
    # No independent watchdog can race the intentional pause for GPU work.
    if (( ${keepalive_paused:-0} == 1 || ${keepalive_stop_requested:-0} == 1 )); then
        return 0
    fi
    find_keepalive_pids
    if (( ${#keepalive_pids[@]} == 0 )); then
        log "CPU protection: keep-alive disappeared; restoring through official controller"
        start_keepalive || return 1
        capture_keepalive_identity || return 1
    elif (( ${#keepalive_pids[@]} != 1 )); then
        log "ERROR: CPU protection found multiple matching keep-alive processes"
        return 1
    elif ! same_keepalive_identity "${keepalive_original_pid}" "${keepalive_original_ticks}"; then
        capture_keepalive_identity || return 1
    fi
    log "CPU protection: verified one active keep-alive while ${preflight_label:-preparing data}"
}

heartbeat_loop() {
    while true; do
        local snapshot
        snapshot="$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory \
            --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
        log "heartbeat variant=${variant}; GPU(index,usedMiB,freeMiB,util%,memutil%)=${snapshot}"
        sleep "${heartbeat_seconds}"
    done
}

write_source_provenance() {
    local destination="$1"
    local source_path relative_path
    {
        printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'git_head=%s\n' "$(git -C "${project_root}" rev-parse HEAD)"
        printf 'git_branch=%s\n' "$(git -C "${project_root}" branch --show-current)"
        printf 'git_status_short_begin\n'
        git -C "${project_root}" status --short
        printf 'git_status_short_end\n'
        for source_path in \
            "${project_root}/run_full_b200.sh" \
            "${project_root}/src/grpo_validate_b200_smoke.py" \
            "${project_root}/src/grpo_validate_b200_dataset.py" \
            "${project_root}/src/grpo_validate_b200_runtime.py" \
            "${project_root}/src/grpo_write_model_sha256_manifest.py" \
            "${project_root}/src/grpo_prepare_dataset.py" \
            "${project_root}/src/grpo_verify_media_decode.py" \
            "${project_root}/src/grpo_media_cache.py" \
            "${project_root}/src/grpo_media_prefetch.py" \
            "${project_root}/src/qwen-vl-utils/src/qwen_vl_utils/vision_process.py" \
            "${project_root}/src/r1-v/src/open_r1/grpo.py" \
            "${project_root}/src/r1-v/src/open_r1/trainer/grpo_trainer.py" \
            "${project_root}/src/r1-v/src/open_r1/trainer/qwen25vl_fa2_compat.py" \
            "${project_root}/src/r1-v/src/open_r1/trainer/video_ktr_grpo_trainer.py" \
            "${project_root}/src/r1-v/src/open_r1/trainer/ktr_token_utils.py"; do
            [[ -f "${source_path}" ]] || continue
            relative_path="${source_path#${project_root}/}"
            printf 'source_sha256[%s]=%s\n' "${relative_path}" "$(sha256sum "${source_path}" | awk '{print $1}')"
        done
        printf 'model_integrity_manifest=%s\n' "${model_integrity_manifest}"
        printf 'model_integrity_manifest_sha256=%s\n' "$(sha256sum "${model_integrity_manifest}" | awk '{print $1}')"
        printf 'environment_manifest=%s\n' "${environment_manifest}"
        printf 'environment_manifest_sha256=%s\n' "$(sha256sum "${environment_manifest}" | awk '{print $1}')"
        "${python_bin}" - <<'PY'
import av
import deepspeed
import flash_attn
import torch
import tokenizers
import transformers
import triton
import triton_kernels
import trl
import torchvision

print(f"python_runtime={__import__('sys').version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"torchvision={torchvision.__version__}")
print(f"av={av.__version__}")
print(f"deepspeed={deepspeed.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"transformers={transformers.__version__}")
print(f"tokenizers={tokenizers.__version__}")
print(f"trl={trl.__version__}")
print(f"triton={triton.__version__}")
PY
    } > "${destination}"
}

write_run_failure() {
    local status="$1"
    [[ -n "${training_started}" ]] || return 0
    [[ -f "${run_root}/training_failed.json" ]] && return 0
    "${python_bin}" - "${run_root}/training_failed.json" "${status}" "${variant}" "${training_dataset_json:-}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

destination, status, variant, dataset = sys.argv[1:]
Path(destination).write_text(
    json.dumps(
        {
            "event": "training_failed",
            "ended_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "exit_code": int(status),
            "variant": variant,
            "training_dataset_json": dataset or None,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

write_summary() {
    [[ -n "${training_started}" && "${summary_written}" != "1" ]] || return 0
    training_ended="${training_ended:-$(date +%s.%N)}"
    training_exit="${training_exit:-1}"
    "${python_bin}" -u "${project_root}/src/grpo_run_summary.py" \
        --run-dir "${run_root}" \
        --resource-log "${run_root}/gpu_metrics.jsonl" \
        --variant "${variant}" \
        --exit-code "${training_exit}" \
        --started-epoch-seconds "${training_started}" \
        --ended-epoch-seconds "${training_ended}" \
        >> "${run_root}/terminal.log" 2>&1
    summary_written=1
}

validate_resource_summary() {
    "${python_bin}" - "${run_root}/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"resource summary is missing: {path}")
summary = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(summary, dict) or summary.get("succeeded") is not True:
    raise SystemExit("resource summary is not a successful training record")
if summary.get("monitor_errors"):
    raise SystemExit(f"GPU monitor reported errors: {summary['monitor_errors']!r}")
gpus = summary.get("gpus")
if not isinstance(gpus, dict):
    raise SystemExit("resource summary has no GPU samples")
expected = {str(index) for index in range(8)}
if set(gpus) != expected:
    raise SystemExit(f"resource summary GPU set mismatch: observed={sorted(gpus)!r}")
missing = [index for index in sorted(expected) if not isinstance(gpus[index], dict) or int(gpus[index].get("samples", 0)) < 1]
if missing:
    raise SystemExit(f"resource summary has no sample for GPU(s): {missing}")
print("[b200-full-resource-gate] passed: eight GPUs sampled with no monitor errors", flush=True)
PY
}

cleanup() {
    local status=$?
    local restore_status=0
    local owned_processes_stopped=1
    trap - EXIT
    trap '' INT TERM HUP
    set +e
    if ! stop_owned_process_group "CPU preflight (${preflight_label:-unknown})" "${preflight_pid}"; then
        owned_processes_stopped=0
    fi
    if ! stop_owned_process_group "full torchrun" "${training_pid}"; then
        owned_processes_stopped=0
    fi
    stop_child "${heartbeat_pid}"
    stop_child "${monitor_pid}"
    if [[ -n "${training_started}" && "${summary_written}" != "1" ]]; then
        training_exit="${training_exit:-${status}}"
        write_summary
    fi
    if [[ -n "${training_started}" && "${status}" != "0" ]]; then
        write_run_failure "${status}"
    fi
    # A controller may report an error after having stopped its workload.  In
    # that narrow case inspect the exact main process before deciding whether
    # restoration is needed; do not treat a stop request as proof of a pause.
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
    if (( ${keepalive_managed:-0} == 1 && keepalive_paused == 0 )); then
        find_keepalive_pids
        if (( ${#keepalive_pids[@]} == 0 )); then
            keepalive_paused=1
            log "cleanup: protection is absent after CPU work; restoration is required"
        fi
    fi
    if (( keepalive_paused == 1 )); then
        if (( owned_processes_stopped != 1 )); then
            restore_status=1
            log "CRITICAL: keep-alive remains paused because an owned process may still hold GPUs"
        elif ! wait_for_gpu_quiescence "before keep-alive restore"; then
            restore_status=1
            log "CRITICAL: keep-alive remains paused because a compute process still owns a GPU"
        else
            log "cleanup: run ended (exit=${status}); restoring keep-alive through official controller"
            if ! start_keepalive; then
                restore_status=1
                log "CRITICAL: automatic keep-alive restoration failed"
            fi
        fi
    fi
    if (( restore_status != 0 )); then
        log "RECOVERY (only after confirming all training processes stopped): KEEP_ALIVE_DASHBOARD=0 bash ${keepalive_launcher} start"
        status=5
    fi
    if (( status == 0 )); then
        log "B200 full wrapper completed successfully"
    else
        log "B200 full wrapper ended with exit=${status}; artifacts remain at ${run_root}"
    fi
    sleep 0.3
    stop_child "${log_follower_pid}"
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
trap 'exit 129' HUP

run_logged_preflight() {
    local label="$1"
    shift
    if [[ -n "${preflight_pid}" ]]; then
        log "ERROR: cannot start ${label}; tracked preflight ${preflight_label} is still present"
        return 2
    fi
    log "${label}: started; progress remains visible below"
    setsid "$@" >> "${run_root}/terminal.log" 2>&1 &
    preflight_pid="$!"
    preflight_label="${label}"
    local observed_pgid=""
    for (( attempt = 0; attempt < 30; attempt += 1 )); do
        observed_pgid="$(ps -o pgid= -p "${preflight_pid}" 2>/dev/null | tr -d '[:space:]' || true)"
        if [[ "${observed_pgid}" == "${preflight_pid}" ]] || ! process_is_live "${preflight_pid}"; then
            break
        fi
        sleep 0.1
    done
    if process_is_live "${preflight_pid}" && [[ "${observed_pgid}" != "${preflight_pid}" ]]; then
        log "ERROR: ${label} did not receive an isolated process group"
        stop_owned_process_group "CPU preflight (${label})" "${preflight_pid}" || true
        preflight_pid=""
        preflight_label=""
        return 2
    fi
    local status=0
    if (( ${keepalive_managed:-0} == 1 && ${keepalive_paused:-0} == 0 && ${keepalive_stop_requested:-0} == 0 )); then
        local guard_elapsed=0
        while process_is_live "${preflight_pid}"; do
            if (( guard_elapsed % heartbeat_seconds == 0 )); then
                if ! ensure_cpu_keepalive; then
                    log "ERROR: CPU preflight lost managed GPU protection"
                    return 5
                fi
            fi
            sleep 1
            ((guard_elapsed += 1))
        done
    fi
    if wait "${preflight_pid}"; then
        status=0
    else
        status=$?
    fi
    if process_group_is_live "${preflight_pid}"; then
        stop_owned_process_group "CPU preflight (${label}) descendants" "${preflight_pid}" || return 5
    fi
    preflight_pid=""
    preflight_label=""
    if (( status != 0 )); then
        log "ERROR: ${label} failed with exit=${status}"
        return "${status}"
    fi
    log "${label}: passed"
}

log "phase 1/8: validating fixed B200 profile, durable inputs, and controller identity"
if [[ "${variant}" != "baseline" && "${variant}" != "ktr" ]]; then
    log "ERROR: VARIANT must be baseline or ktr; got ${variant}"
    exit 2
fi
if [[ "${allow_pause}" != "1" ]]; then
    log "ERROR: set B200_ALLOW_PAUSE_KEEPALIVE=1 only when this manually requested full run may pause and restore the known keep-alive"
    exit 2
fi
for value in "${heartbeat_seconds}" "${gpu_sample_seconds}" "${decoder_workers}" "${decoder_threads}" "${dataloader_workers}" "${decoder_timeout_seconds}" "${decoder_progress_every}" "${decoder_status_seconds}" "${ddp_timeout_seconds}"; do
    require_positive_integer "configuration value" "${value}"
done
if [[ "${max_steps}" != "-1" && ! "${max_steps}" =~ ^[1-9][0-9]*$ ]]; then
    log "ERROR: MAX_STEPS must be -1 (the complete epoch) or a positive integer"
    exit 2
fi
if ! full_output_selected_token="$(normalize_bool "FULL_OUTPUT_SELECTED_TOKEN" "${full_output_selected_token}")"; then
    exit 2
fi
if [[ "${full_output_selected_token}" == "true" && "${max_steps}" == "-1" ]]; then
    log "ERROR: refuse an unbounded selected-token JSONL; set a bounded MAX_STEPS first"
    exit 2
fi
if [[ "${allow_reduced}" != "0" && "${allow_reduced}" != "1" ]]; then
    log "ERROR: B200_ALLOW_REDUCED_HOLMES must be 0 or 1"
    exit 2
fi
if [[ "${allow_decoder_filtered}" != "0" && "${allow_decoder_filtered}" != "1" ]]; then
    log "ERROR: B200_ALLOW_DECODER_FILTERED must be 0 or 1"
    exit 2
fi
if [[ -z "${keepalive_main}" || -z "${keepalive_launcher}" ]]; then
    log "ERROR: configure B200_KEEPALIVE_MAIN and B200_KEEPALIVE_LAUNCHER in ${env_file} or the environment"
    exit 2
fi
for path in "${python_bin}" "${model_path}" "${data_root}" "${dataset_source}" "${smoke_root}" "${environment_manifest}" "${model_integrity_manifest}" "${keepalive_main}" "${keepalive_launcher}"; do
    require_path "required B200 input" "${path}"
done
for path in "${model_path}" "${data_root}" "${dataset_source}" "${smoke_root}" "${environment_manifest}" "${model_integrity_manifest}"; do
    if is_ephemeral_path "$(realpath -e "${path}")"; then
        log "ERROR: required input must not resolve under /tmp or /mnt: ${path}"
        exit 2
    fi
done
keepalive_main="$(realpath -e "${keepalive_main}")"
keepalive_launcher="$(realpath -e "${keepalive_launcher}")"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! command -v setsid >/dev/null 2>&1; then
    log "ERROR: nvidia-smi and setsid are both required on the B200 node"
    exit 2
fi
gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d '[:space:]')"
if [[ "${gpu_count}" != "8" ]] || nvidia-smi --query-gpu=name --format=csv,noheader | grep -Fv 'B200' >/dev/null; then
    log "ERROR: this profile requires exactly eight visible B200 GPUs"
    exit 2
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
site_dir="$("${python_bin}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
require_path "B200 venv site-packages" "${site_dir}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VIDEO_KTR_MEDIA_CACHE_DIR="${media_cache_dir}"
export FORCE_QWENVL_VIDEO_READER=torchvision
export SETUPTOOLS_USE_DISTUTILS=local
export PYTHONPATH="${site_dir}:${project_root}/src:${project_root}/src/r1-v/src/open_r1:${project_root}/src/r1-v/src:${project_root}/src/qwen-vl-utils/src"
# torch.distributed.run honors an inherited PYTHON_EXEC for its workers.  A
# platform-level override can silently launch /opt/venv workers even though
# the torchrun parent and runtime gate use our pinned project venv.
export PYTHON_EXEC="${python_bin}"
export MODEL_PATH="${model_path}"

keepalive_managed=1
find_keepalive_pids
if (( ${#keepalive_pids[@]} == 0 )); then
    start_keepalive
elif (( ${#keepalive_pids[@]} != 1 )); then
    log "ERROR: multiple matching keep-alive processes; controller identity is ambiguous"
    exit 2
fi

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
    log "ERROR: Triton requires CUDA headers; set TRITON_CUDA_INCLUDE_DIR"
    exit 2
fi
triton_cuda_include_dir="$(realpath -e "${triton_cuda_include_dir}")"
if ! command -v gcc >/dev/null 2>&1 || ! printf '#include <cuda.h>\n' | gcc -I"${triton_cuda_include_dir}" -x c -E -o /dev/null -; then
    log "ERROR: GCC cannot preprocess the selected CUDA headers"
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
    log "ERROR: Triton requires an executable ptxas; set TRITON_PTXAS_PATH"
    exit 2
fi
triton_ptxas_path="$(realpath -e "${triton_ptxas_path}")"
if ! "${triton_ptxas_path}" --version | grep -Eq 'release [0-9]+\.[0-9]+'; then
    log "ERROR: selected ptxas does not report a usable CUDA release: ${triton_ptxas_path}"
    exit 2
fi
export TRITON_PTXAS_PATH="${triton_ptxas_path}"
triton_tmpdir="${B200_TRITON_TMPDIR:-${b200_root}/.triton-tmp}"
triton_cache_dir="${B200_TRITON_CACHE_DIR:-${b200_root}/.triton-cache}"
for triton_path in "${triton_tmpdir}" "${triton_cache_dir}"; do
    if [[ "${triton_path}" != /* ]] || is_ephemeral_path "${triton_path}"; then
        log "ERROR: Triton cache/temp must not use /tmp or /mnt: ${triton_path}"
        exit 2
    fi
    mkdir -p "${triton_path}"
done
export TMPDIR="${triton_tmpdir}"
export TRITON_CACHE_DIR="${triton_cache_dir}"

log "phase 2/8: hashing the pinned model, checking environment drift, FA2 sm_100 binary, and separate image/video token IDs"
run_logged_preflight "B200 pinned model/environment integrity gate" \
    env CUDA_VISIBLE_DEVICES="" \
    "${python_bin}" -u "${project_root}/src/grpo_validate_b200_runtime.py" \
    --model-path "${model_path}" \
    --model-integrity-manifest "${model_integrity_manifest}" \
    --environment-manifest "${environment_manifest}" \
    --project-root "${project_root}" \
    --output "${runtime_gate}"
"${python_bin}" - >> "${run_root}/terminal.log" 2>&1 <<'PY'
import os
from importlib import metadata

import flash_attn
import torch
import tokenizers
import transformers
import triton
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
image_id, video_id = int(config.image_token_id), int(config.video_token_id)
if image_id == video_id:
    raise SystemExit(f"Qwen image/video token IDs must differ, both={image_id}")
print(
    "[b200-full-preflight] "
    f"torch={torch.__version__}; cuda={torch.version.cuda}; flash_attn={flash_attn.__version__}; "
    f"transformers={transformers.__version__}; image_token_id={image_id}; video_token_id={video_id}",
    flush=True,
)
PY
flash_binary="$("${python_bin}" -c 'import torch; import flash_attn_2_cuda; print(flash_attn_2_cuda.__file__)')"
if [[ -z "${flash_binary}" ]] || ! "${python_bin}" - "${flash_binary}" <<'PY'
from pathlib import Path
import sys

payload = Path(sys.argv[1]).read_bytes().lower()
raise SystemExit(0 if b"sm_100" in payload or b"sm100" in payload or b"compute_100" in payload else 1)
PY
then
    log "ERROR: installed FlashAttention-2 binary lacks an sm_100 marker; refusing fallback"
    exit 2
fi
log "FA2 B200 binary and runtime bridge passed; Triton cache=${triton_cache_dir}"
run_logged_preflight "eight-rank CPU-only training-entrypoint import gate" \
    env CUDA_VISIBLE_DEVICES="" \
    "${python_bin}" -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    --log-dir "${run_root}/import-gate-logs" \
    --tee 3 \
    "${project_root}/src/grpo_validate_b200_training_import.py" \
    --expected-python "${python_bin}" \
    --entrypoint "${project_root}/src/r1-v/src/open_r1/grpo.py"

log "phase 3/8: validating the passed B200 smoke and current critical training-source hashes"
run_logged_preflight "B200 smoke provenance gate" \
    "${python_bin}" -u "${project_root}/src/grpo_validate_b200_smoke.py" \
    --smoke-root "${smoke_root}" \
    --project-root "${project_root}" \
    --output "${smoke_gate}"

log "phase 4/8: ensuring the external keep-alive is active during CPU-only data preparation"
find_keepalive_pids
if (( ${#keepalive_pids[@]} == 0 )); then
    log "keep-alive is absent; starting it through its official controller before CPU preflight"
    start_keepalive
elif (( ${#keepalive_pids[@]} != 1 )); then
    log "ERROR: expected exactly one known keep-alive main process, observed ${#keepalive_pids[@]}"
    exit 2
else
    log "keep-alive preflight: exactly one known main process is active"
fi
capture_keepalive_identity

log "phase 5/8: path-verifying the mixed Holmes-16k data while keep-alive remains active"
run_logged_preflight "mixed path verification" \
    "${python_bin}" -u "${project_root}/src/grpo_prepare_dataset.py" \
    --source "${dataset_source}" \
    --video-root "${data_root}" \
    --output "${path_dataset}" \
    --data-type all \
    --progress-every 1000
path_gate_args=(
    "${python_bin}" -u "${project_root}/src/grpo_validate_b200_dataset.py"
    --source "${dataset_source}"
    --path-dataset "${path_dataset}"
    --path-manifest "${path_manifest}"
    --output "${path_gate}"
)
if [[ "${allow_reduced}" == "1" ]]; then
    path_gate_args+=(--allow-reduced-holmes)
fi
run_logged_preflight "Holmes strict/reduced class gate" "${path_gate_args[@]}"

log "phase 6/8: CPU-only mixed image/video Qwen decode preflight; reuse exact cache=${media_cache_dir}; workers=${decoder_workers}, threads=${decoder_threads}, timeout=${decoder_timeout_seconds}s; first build decodes unique media, later runs validate cached payloads"
decoder_strict_args=()
if [[ "${allow_decoder_filtered}" == "0" ]]; then
    decoder_strict_args+=(--require-all)
fi
run_logged_preflight "mixed media decode verification" \
    env CUDA_VISIBLE_DEVICES="" \
    "${python_bin}" -u "${project_root}/src/grpo_verify_media_decode.py" \
    --source "${path_dataset}" \
    --media-root "${data_root}" \
    --output "${decoder_dataset}" \
    --nframes "${nframes}" \
    --max-pixels "${max_pixels}" \
    --reader-backend torchvision \
    --workers "${decoder_workers}" \
    --threads-per-worker "${decoder_threads}" \
    --cache-dir "${media_cache_dir}" \
    "${decoder_strict_args[@]}" \
    --timeout-seconds "${decoder_timeout_seconds}" \
    --progress-every "${decoder_progress_every}" \
    --status-seconds "${decoder_status_seconds}"
data_gate_args=(
    "${python_bin}" -u "${project_root}/src/grpo_validate_b200_dataset.py"
    --source "${dataset_source}"
    --path-dataset "${path_dataset}"
    --path-manifest "${path_manifest}"
    --decoder-dataset "${decoder_dataset}"
    --decoder-manifest "${decoder_manifest}"
    --nframes "${nframes}"
    --max-pixels "${max_pixels}"
    --output "${data_gate}"
)
if [[ "${allow_reduced}" == "1" ]]; then
    data_gate_args+=(--allow-reduced-holmes)
fi
if [[ "${allow_decoder_filtered}" == "1" ]]; then
    data_gate_args+=(--allow-decoder-filtered)
fi
run_logged_preflight "mixed decoder data-class gate" "${data_gate_args[@]}"
training_dataset_json="${decoder_dataset}"
if [[ -n "${B200_RESUME_FROM_CHECKPOINT:-}" ]]; then
    run_logged_preflight "checkpoint resume contract" \
        "${python_bin}" -u "${project_root}/src/grpo_validate_b200_resume.py" \
        --checkpoint "${B200_RESUME_FROM_CHECKPOINT}" --dataset "${training_dataset_json}" \
        --variant "${variant}" --max-steps "${max_steps}" --output "${run_root}/resume_gate.json"
fi
reproduction_class="$("${python_bin}" - "${data_gate}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["reproduction_class"])
PY
)"
decoder_filtered="$("${python_bin}" - "${data_gate}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(str(bool(payload["decoder_contract"]["decoder_filtered"])).lower())
PY
)"
write_source_provenance "${run_root}/source_provenance.txt"
{
    printf 'profile=8xB200-upstream-aligned\nvariant=%s\nreproduction_class=%s\n' "${variant}" "${reproduction_class}"
    printf 'strict_holmes_16k=%s\ndecoder_filtered=%s\n' "$([[ "${reproduction_class}" == "strict-holmes-16k" ]] && printf true || printf false)" "${decoder_filtered}"
    printf 'dataset_source=%s\npath_dataset=%s\ntraining_dataset_json=%s\n' "${dataset_source}" "${path_dataset}" "${training_dataset_json}"
    printf 'smoke_root=%s\nsmoke_gate=%s\nsmoke_gate_pre_gpu=%s\nruntime_gate=%s\nruntime_gate_pre_gpu=%s\ndata_gate=%s\n' \
        "${smoke_root}" "${smoke_gate}" "${smoke_gate_pre_gpu}" "${runtime_gate}" "${runtime_gate_pre_gpu}" "${data_gate}"
    printf 'model_integrity_manifest=%s\nenvironment_manifest=%s\npython_bin=%s\npython_exec=%s\n' \
        "${model_integrity_manifest}" "${environment_manifest}" "${python_bin}" "${PYTHON_EXEC}"
    printf 'nproc_per_node=8\nmax_prompt_length=16384\nmax_completion_length=768\nnum_generations=8\n'
    printf 'requested_cli_max_pixels=401408\nqwen_file_video_effective_cap=105369\nobserved_smoke_video_pixels=94080-98784\nnframes=8\n'
    printf 'media_decode_workers=%s\nmedia_decode_timeout_seconds=%s\n' "${decoder_workers}" "${decoder_timeout_seconds}"
    printf 'media_cache_dir=%s\nmedia_decode_threads=%s\ndataloader_workers=%s\ndataloader_prefetch_factor=2\n' "${media_cache_dir}" "${decoder_threads}" "${dataloader_workers}"
    printf 'attn_implementation=flash_attention_2\nqwen_fa2_rotary_dtype_compat=true\n'
    printf 'selection_mode=paper\nselection_scope=per_completion\nselection_ratio=0.2\ndelta=absolute\n'
    printf 'temporal_permutations=1\ntemporal_include_reverse=false\ntemporal_seed=43\n'
    printf 'image_video_token_ids=distinct\nvideo_reader_backend=torchvision\nmixed_modality_sampler=enabled\n'
    printf 'max_steps=%s\nsave_steps=100\nsave_total_limit=2\n' "${max_steps}"
    printf 'triton_cuda_include_dir=%s\ntriton_ptxas_path=%s\ntriton_cache_dir=%s\ntriton_tmpdir=%s\n' "${triton_cuda_include_dir}" "${triton_ptxas_path}" "${triton_cache_dir}" "${triton_tmpdir}"
} > "${run_root}/launch_config.txt"
log "data class=${reproduction_class}; decoder_filtered=${decoder_filtered}; source and decoder manifests are frozen under this run root"

# CPU data preparation can take substantially longer than a smoke. Recheck
# the clean source lineage and the full model/runtime identity at the exact
# boundary before the controller is allowed to pause the GPU keep-alive.
log "phase 6.5/8: re-running smoke/source-clean and pinned model/environment gates immediately before GPU work"
run_logged_preflight "B200 smoke provenance recheck before GPU pause" \
    "${python_bin}" -u "${project_root}/src/grpo_validate_b200_smoke.py" \
    --smoke-root "${smoke_root}" \
    --project-root "${project_root}" \
    --output "${smoke_gate_pre_gpu}"
run_logged_preflight "B200 model/environment recheck before GPU pause" \
    env CUDA_VISIBLE_DEVICES="" \
    "${python_bin}" -u "${project_root}/src/grpo_validate_b200_runtime.py" \
    --model-path "${model_path}" \
    --model-integrity-manifest "${model_integrity_manifest}" \
    --environment-manifest "${environment_manifest}" \
    --project-root "${project_root}" \
    --output "${runtime_gate_pre_gpu}"

log "phase 7/8: pausing keep-alive only for B200 GPU work, then checking real FA2/Qwen rotary"
if ! same_keepalive_identity "${keepalive_original_pid}" "${keepalive_original_ticks}"; then
    log "ERROR: keep-alive identity changed before official stop; refusing to manage it"
    exit 2
fi
keepalive_stop_requested=1
KEEP_ALIVE_DASHBOARD=0 bash "${keepalive_launcher}" stop 9>&- 2>&1 | tee -a "${run_root}/keepalive.log" >> "${run_root}/terminal.log"
wait_for_keepalive_count 0 "pause"
keepalive_paused=1
wait_for_gpu_quiescence "after keep-alive pause"
for physical_gpu in 0 1 2 3 4 5 6 7; do
    free_mib="$(nvidia-smi --id="${physical_gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < min_free_gib * 1024 )); then
        log "ERROR: GPU ${physical_gpu} has ${free_mib:-unknown} MiB free; need at least $((min_free_gib * 1024)) MiB"
        exit 2
    fi
done
export VIDEO_KTR_REQUIRE_ATTN_IMPLEMENTATION=flash_attention_2
export VIDEO_KTR_QWEN_FA2_ROTARY_DTYPE_COMPAT=1
export VIDEO_KTR_RANK_TRACE="${B200_RANK_PHASE_TRACE:-0}"
unset TORCH_NCCL_TRACE_BUFFER_SIZE
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-COLL}"
"${python_bin}" - >> "${run_root}/terminal.log" 2>&1 <<'PY'
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
print("[b200-full-preflight] fa2_and_qwen_fp32_rope_rotary_forward_backward=ok", flush=True)
PY
wait_for_gpu_quiescence "after FA2/Qwen rotary gate"

log "phase 8/8: starting 8-rank ${variant} full run; rank-0 progress, GPU heartbeat, and resource JSONL will stream to terminal"
training_started="$(date +%s.%N)"
"${python_bin}" -u "${project_root}/src/grpo_resource_monitor.py" \
    --output "${run_root}/gpu_metrics.jsonl" \
    --interval-seconds "${gpu_sample_seconds}" \
    --run-label "${variant}" \
    > "${run_root}/resource_monitor.log" 2>&1 &
monitor_pid="$!"
sleep 1
if ! process_is_live "${monitor_pid}"; then
    log "ERROR: GPU resource monitor exited before training started"
    exit 2
fi
heartbeat_loop &
heartbeat_pid="$!"

video_ktr_flag=false
if [[ "${variant}" == "ktr" ]]; then
    video_ktr_flag=true
fi
command=(
    "${python_bin}" -u -m torch.distributed.run
    --standalone
    --nproc_per_node=8
    --log-dir "${run_root}/torchrun-logs"
    --tee 3
    --local-ranks-filter 0
    "${project_root}/src/r1-v/src/open_r1/grpo.py"
    --output_dir "${run_root}/training"
    --model_name_or_path "${model_path}"
    --dataset_name "${training_dataset_json}"
    --dataset_train_split train
    --deepspeed "${project_root}/src/r1-v/local_scripts/zero3.json"
    --max_steps "${max_steps}"
    --num_train_epochs 1
    --max_prompt_length 16384
    --max_completion_length 768
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --num_generations 8
    --learning_rate 2e-6
    --lr_scheduler_type cosine
    --weight_decay 0.01
    --bf16 true
    --gradient_checkpointing true
    --attn_implementation flash_attention_2
    --max_pixels 401408
    --min_pixels 3136
    --nframes 8
    --temporal false
    --len_control true
    --beta 0.04
    --max_grad_norm 5
    --logging_steps 1
    --logging_first_step true
    --save_strategy steps
    --save_steps "${save_steps}"
    --save_total_limit "${save_total_limit}"
    --save_only_model false
    --skip_final_model_save false
    --ddp_timeout "${ddp_timeout_seconds}"
    --ds3_gather_for_generation true
    --report_to none
    --remove_unused_columns false
    --dataloader_num_workers "${dataloader_workers}"
    --dataloader_prefetch_factor 2
    --dataloader_persistent_workers true
    --use_vllm false
    --data_root "${data_root}"
    --selection_mode paper
    --selection_scope per_completion
    --temporal_permutations 1
    --temporal_include_reverse false
    --temporal_seed 43
    --video_ktr "${video_ktr_flag}"
    --entropy_ratio 0.2
    --visual_ratio 0.2
    --temporal_ratio 0.2
    --output_selected_token "${full_output_selected_token}"
    --token_record_limit "${token_record_limit}"
    --run_name "Video-KTR-B200-${variant}"
    --seed 42
)
if [[ -n "${B200_RESUME_FROM_CHECKPOINT:-}" ]]; then
    command+=(--resume_from_checkpoint "${B200_RESUME_FROM_CHECKPOINT}")
    log "restoring optimizer, scheduler, RNG and step from ${B200_RESUME_FROM_CHECKPOINT}; keeping original base/reference model"
fi
set +e
B200_TRAINING_LOG="${run_root}/training.log" B200_PIPELINE_LOG="${run_root}/terminal.log" \
    setsid bash -c 'set -o pipefail; "$@" 2>&1 | tee -a "${B200_TRAINING_LOG}" >> "${B200_PIPELINE_LOG}"' bash "${command[@]}" &
training_pid="$!"
for (( attempt = 0; attempt < 30; attempt += 1 )); do
    observed_pgid="$(ps -o pgid= -p "${training_pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "${observed_pgid}" == "${training_pid}" ]] || ! process_is_live "${training_pid}"; then
        break
    fi
    sleep 0.1
done
if process_is_live "${training_pid}" && [[ "${observed_pgid}" != "${training_pid}" ]]; then
    log "ERROR: full torchrun did not receive an isolated process group"
    exit 2
fi
wait "${training_pid}"
training_exit="$?"
set -e
if process_group_is_live "${training_pid}"; then
    stop_owned_process_group "full torchrun descendants" "${training_pid}" || exit 5
fi
training_pid=""
training_ended="$(date +%s.%N)"
stop_child "${heartbeat_pid}"
stop_child "${monitor_pid}"
heartbeat_pid=""
monitor_pid=""
log "writing duration, memory, utilization, and final-step summary"
write_summary
if (( training_exit != 0 )); then
    log "full ${variant} training failed with exit=${training_exit}"
    for rank_log in "${run_root}"/torchrun-logs/*/attempt_*/*/stderr.log; do
        if [[ -s "${rank_log}" ]] && grep -Eiq 'Traceback|Error|Exception' "${rank_log}"; then
            log "first worker traceback: ${rank_log}"
            tail -n 80 "${rank_log}" >> "${run_root}/terminal.log"
            break
        fi
    done
    exit "${training_exit}"
fi
if [[ ! -f "${run_root}/training/training_complete.json" ]]; then
    log "ERROR: torchrun exited zero but training_complete.json is absent"
    exit 4
fi
validate_resource_summary
log "training PASS; EXIT cleanup will now restore and verify the keep-alive"
