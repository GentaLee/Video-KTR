#!/usr/bin/env bash
# Manual full-data launcher for one direct GRPO variant on 4xH200.
#
# Default is Video-KTR (`VARIANT=ktr`).  For a like-for-like baseline, launch a
# separate run with `VARIANT=baseline`, then call src/grpo_compare_runs.py on
# the two artifact directories.  This script intentionally never selects the
# vLLM path: it does not apply the E union V union T loss mask.

set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
variant="${VARIANT:-ktr}"
output_root="${OUTPUT_ROOT:-${project_root}/artifacts/grpo-full}"
run_root="${RUN_ROOT:-${output_root}/${variant}-${timestamp}}"

python_bin="${PYTHON_BIN:-/usr/bin/python3.12}"
site_dir="${GRPO_SITE_PACKAGES:-${project_root}/.python-packages-grpo}"
model_path="${MODEL_PATH:-}"
data_root="${DATA_ROOT:-}"
dataset_source="${DATASET_SOURCE:-}"
dataset_json="${PREPARED_DATASET:-${run_root}/Video-R1-260k-video-path-verified.json}"
deepspeed_config="${DEEPSPEED_CONFIG:-${project_root}/src/r1-v/local_scripts/zero3.json}"
trainer_program="${project_root}/src/r1-v/src/open_r1/grpo.py"

cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
expected_gpu_name="${EXPECTED_GPU_NAME:-H200}"
min_free_gib="${MIN_FREE_GIB:-120}"
max_prompt_length="${MAX_PROMPT_LENGTH:-4096}"
max_completion_length="${MAX_COMPLETION_LENGTH:-512}"
num_generations="${NUM_GENERATIONS:-4}"
max_pixels="${MAX_PIXELS:-100352}"
nframes="${NFRAMES:-8}"
temporal_permutations="${TEMPORAL_PERMUTATIONS:-5}"
max_steps="${MAX_STEPS:--1}"
save_steps="${SAVE_STEPS:-500}"
save_total_limit="${SAVE_TOTAL_LIMIT:-2}"
heartbeat_seconds="${HEARTBEAT_SECONDS:-30}"
gpu_sample_seconds="${GPU_SAMPLE_SECONDS:-5}"
attn_implementation="${ATTN_IMPLEMENTATION:-sdpa}"
resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-}"
prepared_dataset_only="${USE_EXISTING_PREPARED_DATASET:-0}"

# Full data would produce an unbounded token JSONL if examples were written on
# every optimizer step.  The smoke run already captures CoT examples; leave
# this disabled by default.  An operator may explicitly enable it for a
# bounded MAX_STEPS debugging run only.
full_output_selected_token="${FULL_OUTPUT_SELECTED_TOKEN:-0}"
token_record_limit="${TOKEN_RECORD_LIMIT:-0}"

# This exact keep-alive was external to the task.  The launcher refuses to
# overlap it by default so performance telemetry remains meaningful.  After
# the operator opts in, only the exact verified PID is TERM'd; cleanup starts
# the original launcher again and never uses pkill or SIGKILL.
allow_pause_external_keepalive="${ALLOW_PAUSE_EXTERNAL_KEEPALIVE:-0}"
keepalive_main="${KEEPALIVE_MAIN:-}"
keepalive_launcher="${KEEPALIVE_LAUNCHER:-}"
keepalive_dir=""
keepalive_configured=0
if [[ -n "${keepalive_main}" && -n "${keepalive_launcher}" ]]; then
    keepalive_dir="$(dirname "${keepalive_launcher}")"
    keepalive_configured=1
fi
keepalive_pids=()
external_keepalive_paused=0
paused_keepalive_pid=""
paused_keepalive_ticks=""
monitor_pid=""
heartbeat_pid=""
training_started=""
training_ended=""
training_exit=""
training_pid=""
summary_written=0

mkdir -p "${run_root}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] [grpo-full] %s\n' -1 "$*" | tee -a "${run_root}/terminal.log"
}

stop_child() {
    local child_pid="${1:-}"
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -TERM "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
}

process_is_live() {
    local pid="${1:-}"
    local state
    [[ -n "${pid}" && -r "/proc/${pid}/stat" ]] || return 1
    state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
    [[ -n "${state}" && "${state}" != "Z" ]]
}

# The trainer is deliberately started as a tracked background process below.
# This makes a TERM sent only to this launcher safe: stop torchrun before a
# cleanup handler can restore the external all-GPU keep-alive workload.
stop_training() {
    local pid="${training_pid:-}"
    local waited=0
    if [[ -z "${pid}" ]]; then
        return 0
    fi
    if process_is_live "${pid}"; then
        log "interrupt cleanup: sending TERM to tracked torchrun pid=${pid} before restoring keep-alive"
        kill -TERM "${pid}" 2>/dev/null || true
        while process_is_live "${pid}"; do
            if (( waited >= 45 )); then
                log "CRITICAL: torchrun pid=${pid} did not stop within 45s; keep-alive will remain paused to avoid a GPU conflict"
                return 1
            fi
            sleep 1
            ((waited += 1))
        done
    fi
    wait "${pid}" 2>/dev/null || true
    training_pid=""
    return 0
}

process_start_ticks() {
    local pid="$1"
    [[ -r "/proc/${pid}/stat" ]] || return 1
    awk '{print $22}' "/proc/${pid}/stat"
}

process_command_line() {
    local pid="$1"
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline"
}

find_external_keepalive() {
    keepalive_pids=()
    if (( keepalive_configured != 1 )); then
        return 0
    fi
    local proc pid command_line
    for proc in /proc/[0-9]*; do
        [[ -r "${proc}/cmdline" ]] || continue
        pid="${proc##*/}"
        command_line="$(process_command_line "${pid}" 2>/dev/null || true)"
        if [[ "${command_line}" == *"${keepalive_main}"* ]]; then
            keepalive_pids+=("${pid}")
        fi
    done
}

same_keepalive_identity() {
    local pid="$1"
    local ticks="$2"
    (( keepalive_configured == 1 )) || return 1
    [[ "$(process_start_ticks "${pid}" 2>/dev/null || true)" == "${ticks}" ]] || return 1
    [[ "$(process_command_line "${pid}" 2>/dev/null || true)" == *"${keepalive_main}"* ]]
}

restore_external_keepalive() {
    if (( external_keepalive_paused != 1 )); then
        return 0
    fi
    find_external_keepalive
    if (( ${#keepalive_pids[@]} > 0 )); then
        log "keep-alive restore: an exact main.py process is already present (pid=${keepalive_pids[*]}); no duplicate started"
        external_keepalive_paused=0
        return 0
    fi
    if [[ ! -f "${keepalive_launcher}" ]]; then
        log "CRITICAL: keep-alive was paused but its launcher disappeared: ${keepalive_launcher}"
        return 1
    fi
    log "keep-alive restore: restarting the original launcher ${keepalive_launcher}"
    local new_pid waited=0
    new_pid="$(
        cd "${keepalive_dir}"
        nohup env -u KEEP_ALIVE_GPUS -u KEEP_ALIVE_NUM_GPUS -u KEEP_ALIVE_GPU_COUNT \
            bash "${keepalive_launcher}" \
            >> "${run_root}/keepalive_restart.log" 2>&1 < /dev/null &
        printf '%s' "$!"
    )"
    while (( waited < 30 )); do
        sleep 1
        find_external_keepalive
        if (( ${#keepalive_pids[@]} == 1 )); then
            log "keep-alive restore: verified pid=${keepalive_pids[0]} (launcher_pid=${new_pid}; waited=${waited}s)"
            external_keepalive_paused=0
            return 0
        fi
        if (( ${#keepalive_pids[@]} > 1 )); then
            log "CRITICAL: keep-alive restart became ambiguous; launcher_pid=${new_pid}; observed=${keepalive_pids[*]}"
            return 1
        fi
        ((waited += 1))
    done
    log "CRITICAL: keep-alive restart did not produce the expected process within 30s; launcher_pid=${new_pid}"
    return 1
}

write_summary() {
    if [[ -z "${training_started}" || "${summary_written}" == "1" ]]; then
        return 0
    fi
    training_ended="${training_ended:-$(date +%s.%N)}"
    training_exit="${training_exit:-1}"
    "${python_bin}" -u "${project_root}/src/grpo_run_summary.py" \
        --run-dir "${run_root}" \
        --resource-log "${run_root}/gpu_metrics.jsonl" \
        --variant "${variant}" \
        --exit-code "${training_exit}" \
        --started-epoch-seconds "${training_started}" \
        --ended-epoch-seconds "${training_ended}" \
        2>&1 | tee -a "${run_root}/terminal.log" || true
    summary_written=1
}

print_keepalive_command() {
    if (( keepalive_configured != 1 )); then
        log "external keep-alive management was not configured; no recovery command is available"
        return 0
    fi
    if [[ "${1:-1}" != "1" ]]; then
        log "保活尚未恢复：训练进程仍未确认停止；请先处理该训练进程，切勿立即执行下方命令。"
    fi
    log "保活拉起命令（仅当自动恢复校验失败，且已确认训练已停止、main.py 不存在时使用）："
    log "  bash ${keepalive_launcher}"
}

cleanup() {
    local status=$?
    local training_stopped=1
    trap - EXIT
    set +e
    if ! stop_training; then
        training_stopped=0
    fi
    stop_child "${heartbeat_pid}"
    stop_child "${monitor_pid}"
    heartbeat_pid=""
    monitor_pid=""
    if [[ -n "${training_started}" && "${summary_written}" != "1" ]]; then
        training_exit="${training_exit:-${status}}"
        write_summary
    fi
    if (( training_stopped == 1 )); then
        if ! restore_external_keepalive; then
            log "keep-alive restoration requires immediate operator attention; see ${run_root}/keepalive_restart.log"
        fi
    else
        log "CRITICAL: not restoring keep-alive because tracked training may still own the GPUs"
    fi
    print_keepalive_command "${training_stopped}"
    if (( status == 0 )); then
        log "launcher finished successfully"
    else
        log "launcher stopped with exit=${status}; artifacts remain in ${run_root}"
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

pause_external_keepalive() {
    if (( keepalive_configured != 1 )); then
        log "cannot manage an external keep-alive without both KEEPALIVE_MAIN and KEEPALIVE_LAUNCHER"
        return 3
    fi
    if [[ "${allow_pause_external_keepalive}" != "1" ]]; then
        log "refusing to overlap training with external keep-alive. After explicit approval, rerun with ALLOW_PAUSE_EXTERNAL_KEEPALIVE=1."
        return 3
    fi
    if [[ ! -f "${keepalive_launcher}" || ! -f "${keepalive_main}" ]]; then
        log "cannot safely pause keep-alive: expected files are missing (${keepalive_launcher}, ${keepalive_main})"
        return 3
    fi
    find_external_keepalive
    if (( ${#keepalive_pids[@]} != 1 )); then
        log "cannot safely pause keep-alive: expected exactly one exact process, found ${#keepalive_pids[@]} (${keepalive_pids[*]:-none})"
        return 3
    fi
    paused_keepalive_pid="${keepalive_pids[0]}"
    paused_keepalive_ticks="$(process_start_ticks "${paused_keepalive_pid}")"
    if [[ -z "${paused_keepalive_ticks}" ]] || ! same_keepalive_identity "${paused_keepalive_pid}" "${paused_keepalive_ticks}"; then
        log "cannot safely pause keep-alive: PID identity changed during validation"
        return 3
    fi
    log "keep-alive pause: sending TERM only to verified pid=${paused_keepalive_pid}, start_ticks=${paused_keepalive_ticks}"
    # Take restoration responsibility before signalling.  If the process
    # exits in this narrow interval, EXIT cleanup will recreate it rather
    # than silently losing the external keep-alive workload.
    external_keepalive_paused=1
    if ! kill -TERM "${paused_keepalive_pid}" 2>/dev/null; then
        find_external_keepalive
        if (( ${#keepalive_pids[@]} == 0 )); then
            log "keep-alive exited during pause signalling; cleanup will restore the original launcher"
            return 0
        fi
        external_keepalive_paused=0
        log "cannot safely pause keep-alive: TERM was not delivered and process state is still present (${keepalive_pids[*]})"
        return 3
    fi
    local waited=0
    while same_keepalive_identity "${paused_keepalive_pid}" "${paused_keepalive_ticks}"; do
        if (( waited >= 45 )); then
            log "keep-alive pause timed out; refusing to use SIGKILL or broad process matching"
            return 3
        fi
        sleep 1
        ((waited += 1))
    done
    find_external_keepalive
    if (( ${#keepalive_pids[@]} != 0 )); then
        log "keep-alive pause is ambiguous: a different expected process appeared (${keepalive_pids[*]}); refusing training"
        return 3
    fi
    log "keep-alive pause: verified stopped; cleanup will restart the original launcher"
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

validate_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        log "${name} must be a positive integer; got '${value}'"
        return 2
    fi
}

write_source_provenance() {
    local destination="$1"
    local source_path relative_path
    {
        printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        if git -C "${project_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            printf 'git_head=%s\n' "$(git -C "${project_root}" rev-parse HEAD)"
            printf 'git_branch=%s\n' "$(git -C "${project_root}" branch --show-current)"
            printf 'git_status_short_begin\n'
            git -C "${project_root}" status --short
            printf 'git_status_short_end\n'
            printf 'git_worktree_diff_sha256=%s\n' "$(git -C "${project_root}" diff --no-ext-diff --binary | sha256sum | awk '{print $1}')"
            printf 'git_index_diff_sha256=%s\n' "$(git -C "${project_root}" diff --cached --no-ext-diff --binary | sha256sum | awk '{print $1}')"
        else
            printf 'git_repository=unavailable\n'
        fi
        for source_path in \
            "${project_root}/run_full.sh" \
            "${project_root}/src/grpo_prepare_dataset.py" \
            "${project_root}/src/r1-v/src/open_r1/grpo.py" \
            "${project_root}/src/r1-v/src/open_r1/trainer/grpo_trainer.py" \
            "${project_root}/src/r1-v/src/open_r1/trainer/video_ktr_grpo_trainer.py" \
            "${project_root}/src/r1-v/src/open_r1/trainer/ktr_token_utils.py"; do
            [[ -f "${source_path}" ]] || continue
            relative_path="${source_path#${project_root}/}"
            printf 'source_sha256[%s]=%s\n' "${relative_path}" "$(sha256sum "${source_path}" | awk '{print $1}')"
        done
    } > "${destination}"
}

append_runtime_provenance() {
    local destination="$1"
    "${python_bin}" - <<'PY' >> "${destination}"
import torch
import transformers
import tokenizers
import trl

print(f"python_runtime={__import__('sys').version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"tokenizers={tokenizers.__version__}")
print(f"trl={trl.__version__}")
PY
}

log "phase 1/6: validating 4xH200 capacity, full data, and offline runtime"
if [[ "${variant}" != "baseline" && "${variant}" != "ktr" ]]; then
    log "ERROR: VARIANT must be baseline or ktr; got ${variant}"
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi is unavailable; run this launcher on the GPU node"
    exit 2
fi
if [[ -z "${model_path}" || -z "${data_root}" ]]; then
    log "ERROR: set MODEL_PATH and DATA_ROOT to the local checkpoint and Video-R1 data root before launching"
    exit 2
fi
if [[ -z "${dataset_source}" ]]; then
    dataset_source="${data_root}/Video-R1-260k.json"
fi
if [[ -n "${keepalive_main}" || -n "${keepalive_launcher}" ]]; then
    if (( keepalive_configured != 1 )); then
        log "ERROR: configure both KEEPALIVE_MAIN and KEEPALIVE_LAUNCHER, or neither."
        exit 2
    fi
elif [[ "${allow_pause_external_keepalive}" == "1" ]]; then
    log "ERROR: ALLOW_PAUSE_EXTERNAL_KEEPALIVE=1 requires KEEPALIVE_MAIN and KEEPALIVE_LAUNCHER."
    exit 2
else
    log "external keep-alive management is not configured; no external process will be detected or controlled"
fi
for required_path in "${python_bin}" "${model_path}" "${data_root}" "${dataset_source}" "${deepspeed_config}" "${trainer_program}"; do
    if [[ ! -e "${required_path}" ]]; then
        log "ERROR: required path is unavailable: ${required_path}"
        exit 2
    fi
done
if [[ ! -d "${site_dir}" ]]; then
    log "ERROR: GRPO overlay is unavailable: ${site_dir}; run ./setup_grpo_env.sh on this GPU node first"
    exit 2
fi
for value in "${nproc_per_node}" "${min_free_gib}" "${max_prompt_length}" "${max_completion_length}" "${num_generations}" "${nframes}" "${temporal_permutations}" "${save_steps}" "${save_total_limit}" "${heartbeat_seconds}"; do
    validate_positive_integer "configuration value" "${value}"
done
if [[ "${max_steps}" != "-1" && ! "${max_steps}" =~ ^[1-9][0-9]*$ ]]; then
    log "ERROR: MAX_STEPS must be -1 (one full epoch) or a positive integer; got '${max_steps}'"
    exit 2
fi
if [[ "${nproc_per_node}" != "4" ]]; then
    log "ERROR: this validated full profile is intentionally 4-rank H200; got NPROC_PER_NODE=${nproc_per_node}"
    exit 2
fi
case "${full_output_selected_token}" in
    1|true|TRUE) full_output_selected_token=true ;;
    0|false|FALSE) full_output_selected_token=false ;;
    *) log "ERROR: FULL_OUTPUT_SELECTED_TOKEN must be 0/1 or false/true"; exit 2 ;;
esac
if [[ "${full_output_selected_token}" == "true" && "${max_steps}" == "-1" ]]; then
    log "ERROR: refuse unbounded full token JSONL. Set MAX_STEPS to a bounded debug value before FULL_OUTPUT_SELECTED_TOKEN=1."
    exit 2
fi

IFS=',' read -r -a physical_gpus <<< "${cuda_visible_devices}"
if (( ${#physical_gpus[@]} != nproc_per_node )); then
    log "ERROR: CUDA_VISIBLE_DEVICES=${cuda_visible_devices} exposes ${#physical_gpus[@]} cards, but ${nproc_per_node} are required"
    exit 2
fi
required_mib=$((min_free_gib * 1024))
for physical_gpu in "${physical_gpus[@]}"; do
    if [[ ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
        log "ERROR: CUDA_VISIBLE_DEVICES must contain physical numeric indices; got '${physical_gpu}'"
        exit 2
    fi
    gpu_name="$(nvidia-smi --id="${physical_gpu}" --query-gpu=name --format=csv,noheader | tr -d '\r')"
    free_mib="$(nvidia-smi --id="${physical_gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${gpu_name}" != *"${expected_gpu_name}"* ]]; then
        log "ERROR: GPU ${physical_gpu} is ${gpu_name}; expected a ${expected_gpu_name} profile. Do not silently switch to B200."
        exit 2
    fi
    if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < required_mib )); then
        log "ERROR: GPU ${physical_gpu} has ${free_mib:-unknown} MiB free; need at least ${required_mib} MiB"
        exit 2
    fi
    log "GPU ${physical_gpu}: ${gpu_name}; free=${free_mib} MiB; threshold=${required_mib} MiB"
done

# Make the subsequent Python preflight see exactly the four validated cards.
export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"

if (( keepalive_configured == 1 )); then
    find_external_keepalive
    if (( ${#keepalive_pids[@]} > 0 )); then
        log "detected pre-existing all-GPU keep-alive pid(s): ${keepalive_pids[*]}; it is not part of this launcher"
        pause_external_keepalive
    fi
fi

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${site_dir}:${project_root}/src:${project_root}/src/r1-v/src/open_r1:${project_root}/src/r1-v/src:${project_root}/src/qwen-vl-utils/src${PYTHONPATH:+:${PYTHONPATH}}"
export GRPO_EXPECTED_GPU_COUNT="${nproc_per_node}"
source_provenance="${run_root}/source_provenance.txt"
write_source_provenance "${source_provenance}"

log "phase 2/6: verifying imports before full model loading"
"${python_bin}" - <<'PY' 2>&1 | tee -a "${run_root}/terminal.log"
import os
import torch
import transformers
import tokenizers
import trl
from trainer.video_ktr_grpo_trainer import VideoKTRGRPOTrainer

expected = int(os.environ["GRPO_EXPECTED_GPU_COUNT"])
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
assert transformers.__version__ == "4.49.0.dev0", transformers.__version__
assert tokenizers.__version__ == "0.21.4", tokenizers.__version__
assert trl.__version__ == "0.16.0", trl.__version__
print(
    "[full-preflight] "
    f"torch={torch.__version__}; transformers={transformers.__version__}; "
    f"trl={trl.__version__}; devices={[torch.cuda.get_device_name(i) for i in range(expected)]}; "
    f"trainer={VideoKTRGRPOTrainer.__module__}",
    flush=True,
)
PY
append_runtime_provenance "${source_provenance}"

if [[ "${prepared_dataset_only}" == "1" ]]; then
    if [[ ! -f "${dataset_json}" ]]; then
        log "ERROR: USE_EXISTING_PREPARED_DATASET=1 but dataset is unavailable: ${dataset_json}"
        exit 2
    fi
    log "phase 3/6: reusing prepared path-verified video dataset ${dataset_json}"
else
    log "phase 3/6: filtering all Video-R1 records to path-present, non-empty video samples; progress follows every 5,000 records"
    "${python_bin}" -u "${project_root}/src/grpo_prepare_dataset.py" \
        --source "${dataset_source}" \
        --video-root "${data_root}" \
        --output "${dataset_json}" \
        --data-type video \
        --progress-every 5000 \
        2>&1 | tee -a "${run_root}/terminal.log"
fi

{
    printf 'run_root=%s\nvariant=%s\n' "${run_root}" "${variant}"
    printf 'cuda_visible_devices=%s\nmodel_path=%s\ndata_root=%s\n' "${cuda_visible_devices}" "${model_path}" "${data_root}"
    printf 'dataset_source=%s\ndataset_json=%s\n' "${dataset_source}" "${dataset_json}"
    printf 'max_steps=%s\nmax_prompt_length=%s\nmax_completion_length=%s\nnum_generations=%s\n' "${max_steps}" "${max_prompt_length}" "${max_completion_length}" "${num_generations}"
    printf 'nframes=%s\nmax_pixels=%s\ntemporal_permutations=%s\nattn_implementation=%s\n' "${nframes}" "${max_pixels}" "${temporal_permutations}" "${attn_implementation}"
    printf 'save_steps=%s\nsave_total_limit=%s\nselection_mode=paper\nselection_scope=per_completion\n' "${save_steps}" "${save_total_limit}"
} > "${run_root}/launch_config.txt"

log "phase 4/6: starting full ${variant} training; heartbeat follows every ${heartbeat_seconds}s"
training_started="$(date +%s.%N)"
"${python_bin}" -u "${project_root}/src/grpo_resource_monitor.py" \
    --output "${run_root}/gpu_metrics.jsonl" \
    --interval-seconds "${gpu_sample_seconds}" \
    --run-label "${variant}" \
    > "${run_root}/resource_monitor.log" 2>&1 &
monitor_pid="$!"
heartbeat_loop &
heartbeat_pid="$!"

video_ktr_flag=false
if [[ "${variant}" == "ktr" ]]; then
    video_ktr_flag=true
fi

command=(
    "${python_bin}" -u -m torch.distributed.run --standalone --nproc_per_node="${nproc_per_node}"
    "${trainer_program}"
    --output_dir "${run_root}/training"
    --model_name_or_path "${model_path}"
    --dataset_name "${dataset_json}"
    --dataset_train_split train
    --deepspeed "${deepspeed_config}"
    --max_steps "${max_steps}"
    --num_train_epochs 1
    --max_prompt_length "${max_prompt_length}"
    --max_completion_length "${max_completion_length}"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --num_generations "${num_generations}"
    --learning_rate 2e-6
    --lr_scheduler_type cosine
    --weight_decay 0.01
    --bf16 true
    --gradient_checkpointing true
    --attn_implementation "${attn_implementation}"
    --max_pixels "${max_pixels}"
    --min_pixels 3136
    --nframes "${nframes}"
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
    --report_to none
    --remove_unused_columns false
    --use_vllm false
    --data_root "${data_root}"
    --selection_mode paper
    --selection_scope per_completion
    --temporal_permutations "${temporal_permutations}"
    --temporal_include_reverse true
    --temporal_seed 43
    --video_ktr "${video_ktr_flag}"
    --output_selected_token "${full_output_selected_token}"
    --token_record_limit "${token_record_limit}"
    --run_name "Video-KTR-full-${variant}"
    --seed 42
)
if [[ -n "${resume_from_checkpoint}" ]]; then
    command+=(--resume_from_checkpoint "${resume_from_checkpoint}")
    log "resume requested from ${resume_from_checkpoint}"
fi

set +e
"${command[@]}" > >(tee -a "${run_root}/training.log") 2>&1 &
training_pid="$!"
wait "${training_pid}"
training_exit="$?"
training_pid=""
set -e
training_ended="$(date +%s.%N)"
stop_child "${heartbeat_pid}"
stop_child "${monitor_pid}"
heartbeat_pid=""
monitor_pid=""

log "phase 5/6: writing duration, memory, and utilization summary"
write_summary
if (( training_exit != 0 )); then
    log "full ${variant} training failed with exit=${training_exit}"
    exit "${training_exit}"
fi
if [[ ! -f "${run_root}/training/training_complete.json" ]]; then
    log "full ${variant} training did not produce training_complete.json despite zero launcher exit"
    exit 4
fi

log "phase 6/6: PASS. Inspect ${run_root}/training_summary.md and ${run_root}/training/training_complete.json"
