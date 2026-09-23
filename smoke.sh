#!/usr/bin/env bash
# Run one real optimizer step for baseline GRPO and Video-KTR GRPO on 4xH200.
#
# This launcher is deliberately direct (not vLLM): only the direct trainer
# applies the E union V union T token mask to both GRPO policy loss and KL.
# It is offline-only, emits a terminal heartbeat, and leaves a duration/GPU
# telemetry trail beside every variant.

set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="${OUTPUT_ROOT:-${project_root}/artifacts/grpo-smoke}"
run_root="${RUN_ROOT:-${output_root}/${timestamp}}"

python_bin="${PYTHON_BIN:-/usr/bin/python3.12}"
site_dir="${GRPO_SITE_PACKAGES:-${project_root}/.python-packages-grpo}"
model_path="${MODEL_PATH:-}"
data_root="${DATA_ROOT:-}"
dataset_source="${DATASET_SOURCE:-${project_root}/src/r1-v/video_ktr_data/Video-R1-Holmes-16k.json}"
dataset_ids="${SMOKE_PROBLEM_IDS:-10648,10895,11260,12788}"
deepspeed_config="${DEEPSPEED_CONFIG:-${project_root}/src/r1-v/local_scripts/zero3.json}"
trainer_program="${project_root}/src/r1-v/src/open_r1/grpo.py"

cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
nproc_per_node="${NPROC_PER_NODE:-4}"
expected_gpu_name="${EXPECTED_GPU_NAME:-H200}"
required_nproc_per_node="${REQUIRED_NPROC_PER_NODE:-4}"
smoke_profile="${SMOKE_PROFILE_NAME:-4xH200}"
min_free_gib="${MIN_FREE_GIB:-120}"
max_prompt_length="${MAX_PROMPT_LENGTH:-4096}"
max_completion_length="${MAX_COMPLETION_LENGTH:-256}"
num_generations="${NUM_GENERATIONS:-4}"
max_pixels="${MAX_PIXELS:-100352}"
nframes="${NFRAMES:-8}"
temporal_permutations="${TEMPORAL_PERMUTATIONS:-2}"
temporal_include_reverse="${TEMPORAL_INCLUDE_REVERSE:-true}"
smoke_data_type="${SMOKE_DATA_TYPE:-video}"
token_record_limit="${TOKEN_RECORD_LIMIT:-12}"
heartbeat_seconds="${HEARTBEAT_SECONDS:-20}"
gpu_sample_seconds="${GPU_SAMPLE_SECONDS:-2}"
variant_request="${VARIANT:-both}"
# This GPU image's FlashAttention rotary kernel is incompatible with the
# checkpoint-required Transformers 4.49 mRoPE dtype path.  SDPA is stable for
# the conservative sequence lengths below and avoids changing checkpoint code.
attn_implementation="${ATTN_IMPLEMENTATION:-sdpa}"

# The external keep-alive below existed before this task.  Default behavior is
# to refuse training while it is present, since its all-GPU workload would make
# training-time and memory telemetry invalid.  Set this only after an explicit
# authorization to pause this exact process; cleanup restores it by launching
# the original script and never uses pkill or SIGKILL.
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
paused_keepalive_pid=""
paused_keepalive_ticks=""
external_keepalive_paused=0
monitor_pid=""
heartbeat_pid=""
training_pid=""

mkdir -p "${run_root}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] [grpo-smoke] %s\n' -1 "$*" | tee -a "${run_root}/terminal.log"
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
        return 0
    fi
    if [[ ! -f "${keepalive_launcher}" ]]; then
        log "CRITICAL: keep-alive was paused but its launcher disappeared: ${keepalive_launcher}"
        return 1
    fi
    log "keep-alive restore: restarting the original launcher ${keepalive_launcher}"
    local new_pid
    new_pid="$(
        cd "${keepalive_dir}"
        nohup env -u KEEP_ALIVE_GPUS -u KEEP_ALIVE_NUM_GPUS -u KEEP_ALIVE_GPU_COUNT \
            bash "${keepalive_launcher}" \
            >> "${run_root}/keepalive_restart.log" 2>&1 < /dev/null &
        printf '%s' "$!"
    )"
    local waited=0
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
    if (( training_stopped == 1 )); then
        if ! restore_external_keepalive; then
            log "keep-alive restoration requires immediate operator attention; see ${run_root}/keepalive_restart.log"
        fi
    else
        log "CRITICAL: not restoring keep-alive because tracked training may still own the GPUs"
    fi
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
        log "refusing to overlap training with external keep-alive. To pause it only after explicit approval, rerun with ALLOW_PAUSE_EXTERNAL_KEEPALIVE=1."
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
    local label="$1"
    while true; do
        local snapshot
        snapshot="$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory \
            --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
        log "heartbeat variant=${label}; GPU(index,usedMiB,freeMiB,util%,memutil%)=${snapshot}"
        sleep "${heartbeat_seconds}"
    done
}

stop_variant_children() {
    stop_child "${heartbeat_pid}"
    stop_child "${monitor_pid}"
    heartbeat_pid=""
    monitor_pid=""
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
            "${project_root}/smoke.sh" \
            "${project_root}/src/grpo_prepare_dataset.py" \
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
        printf 'smoke_profile=%s\n' "${smoke_profile}"
        printf 'required_nproc_per_node=%s\n' "${required_nproc_per_node}"
        printf 'smoke_data_type=%s\n' "${smoke_data_type}"
        printf 'temporal_include_reverse=%s\n' "${temporal_include_reverse}"
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

log "phase 1/7: checking ${smoke_profile} capacity, local inputs, and offline runtime"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi is unavailable; run this launcher on the GPU node"
    exit 2
fi
if [[ -z "${model_path}" || -z "${data_root}" ]]; then
    log "ERROR: set MODEL_PATH and DATA_ROOT to the local checkpoint and Video-R1 data root before launching"
    exit 2
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
for value in "${nproc_per_node}" "${required_nproc_per_node}" "${min_free_gib}" "${max_prompt_length}" "${max_completion_length}" "${num_generations}" "${nframes}" "${temporal_permutations}" "${heartbeat_seconds}"; do
    validate_positive_integer "configuration value" "${value}"
done
if [[ "${nproc_per_node}" != "${required_nproc_per_node}" ]]; then
    log "ERROR: ${smoke_profile} requires NPROC_PER_NODE=${required_nproc_per_node}; got ${nproc_per_node}"
    exit 2
fi
if [[ "${smoke_data_type}" != "video" && "${smoke_data_type}" != "image" && "${smoke_data_type}" != "all" ]]; then
    log "ERROR: SMOKE_DATA_TYPE must be video, image, or all; got ${smoke_data_type}"
    exit 2
fi
if [[ "${variant_request}" != "both" && "${variant_request}" != "baseline" && "${variant_request}" != "ktr" ]]; then
    log "ERROR: VARIANT must be both, baseline, or ktr; got ${variant_request}"
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

# Keep the import gate aligned with the exact four validated physical cards.
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

log "phase 2/7: verifying the isolated GRPO imports before model loading"
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
    "[smoke-preflight] "
    f"torch={torch.__version__}; transformers={transformers.__version__}; "
    f"trl={trl.__version__}; visible_gpus={[torch.cuda.get_device_name(i) for i in range(expected)]}; "
    f"trainer={VideoKTRGRPOTrainer.__module__}",
    flush=True,
)
PY
append_runtime_provenance "${source_provenance}"

dataset_json="${run_root}/smoke-${smoke_data_type}-dataset.json"
log "phase 3/7: creating a path-verified ${smoke_data_type} dataset subset (${dataset_ids})"
"${python_bin}" -u "${project_root}/src/grpo_prepare_dataset.py" \
    --source "${dataset_source}" \
    --video-root "${data_root}" \
    --output "${dataset_json}" \
    --data-type "${smoke_data_type}" \
    --problem-ids "${dataset_ids}" \
    2>&1 | tee -a "${run_root}/terminal.log"

{
    printf 'run_root=%s\n' "${run_root}"
    printf 'cuda_visible_devices=%s\n' "${cuda_visible_devices}"
    printf 'model_path=%s\n' "${model_path}"
    printf 'data_root=%s\n' "${data_root}"
    printf 'dataset_source=%s\n' "${dataset_source}"
    printf 'dataset_ids=%s\n' "${dataset_ids}"
    printf 'max_prompt_length=%s\n' "${max_prompt_length}"
    printf 'max_completion_length=%s\n' "${max_completion_length}"
    printf 'num_generations=%s\n' "${num_generations}"
    printf 'nframes=%s\n' "${nframes}"
    printf 'temporal_permutations=%s\n' "${temporal_permutations}"
    printf 'temporal_include_reverse=%s\n' "${temporal_include_reverse}"
    printf 'smoke_profile=%s\nsmoke_data_type=%s\n' "${smoke_profile}" "${smoke_data_type}"
    printf 'attn_implementation=%s\n' "${attn_implementation}"
    printf 'selection_mode=paper\nselection_scope=per_completion\n'
} > "${run_root}/launch_config.txt"

run_variant() {
    local variant="$1"
    local variant_dir="${run_root}/${variant}"
    local training_dir="${variant_dir}/training"
    local resource_log="${variant_dir}/gpu_metrics.jsonl"
    local started ended exit_code
    mkdir -p "${training_dir}"
    log "phase 4/7: ${variant} starting; terminal heartbeat follows every ${heartbeat_seconds}s"
    started="$(date +%s.%N)"
    "${python_bin}" -u "${project_root}/src/grpo_resource_monitor.py" \
        --output "${resource_log}" \
        --interval-seconds "${gpu_sample_seconds}" \
        --run-label "${variant}" \
        > "${variant_dir}/resource_monitor.log" 2>&1 &
    monitor_pid="$!"
    heartbeat_loop "${variant}" &
    heartbeat_pid="$!"

    local -a command=(
        "${python_bin}" -u -m torch.distributed.run --standalone --nproc_per_node="${nproc_per_node}"
        "${trainer_program}"
        --output_dir "${training_dir}"
        --model_name_or_path "${model_path}"
        --dataset_name "${dataset_json}"
        --dataset_train_split train
        --deepspeed "${deepspeed_config}"
        --max_steps 1
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
        --save_strategy no
        --skip_final_model_save true
        --report_to none
        --remove_unused_columns false
        --use_vllm false
        --data_root "${data_root}"
        --selection_mode paper
        --selection_scope per_completion
        --temporal_permutations "${temporal_permutations}"
        --temporal_include_reverse "${temporal_include_reverse}"
        --temporal_seed 43
        --run_name "Video-KTR-${smoke_profile}-${variant}"
        --seed 42
    )
    if [[ -n "${VIDEO_KTR_MEDIA_CACHE_DIR:-}" ]]; then
        command+=(--dataloader_num_workers 2 --dataloader_prefetch_factor 2 --dataloader_persistent_workers true)
    fi
    if [[ "${variant}" == "ktr" ]]; then
        command+=(
            --video_ktr true
            --entropy_ratio 0.2
            --visual_ratio 0.2
            --temporal_ratio 0.2
            --output_selected_token true
            --token_record_limit "${token_record_limit}"
        )
    else
        command+=(
            --video_ktr false
            --output_selected_token false
            --token_record_limit 0
        )
    fi

    set +e
    "${command[@]}" > >(tee -a "${variant_dir}/training.log") 2>&1 &
    training_pid="$!"
    wait "${training_pid}"
    exit_code="$?"
    training_pid=""
    set -e
    ended="$(date +%s.%N)"
    stop_variant_children

    "${python_bin}" -u "${project_root}/src/grpo_run_summary.py" \
        --run-dir "${variant_dir}" \
        --resource-log "${resource_log}" \
        --variant "${variant}" \
        --exit-code "${exit_code}" \
        --started-epoch-seconds "${started}" \
        --ended-epoch-seconds "${ended}" \
        2>&1 | tee -a "${run_root}/terminal.log" || true
    if (( exit_code != 0 )); then
        log "${variant} failed with exit=${exit_code}; no later variant will run"
        return "${exit_code}"
    fi
    if [[ ! -f "${training_dir}/training_complete.json" ]]; then
        log "${variant} did not produce training_complete.json despite a zero launcher exit"
        return 4
    fi
    if [[ "${variant}" == "ktr" ]]; then
        log "phase 5/7: rendering real E/V/T selected-token examples from KTR completions"
        "${python_bin}" -u "${project_root}/src/grpo_token_report.py" \
            --run-dir "${variant_dir}" \
            --examples-per-category 5 \
            --require-all-categories \
            2>&1 | tee -a "${run_root}/terminal.log"
    fi
    log "${variant} PASS: summary=${variant_dir}/training_summary.md"
}

case "${variant_request}" in
    both)
        run_variant baseline
        run_variant ktr
        ;;
    baseline)
        run_variant baseline
        ;;
    ktr)
        run_variant ktr
        ;;
esac

if [[ "${variant_request}" == "both" ]]; then
    log "phase 6/7: writing baseline-vs-KTR time and GPU-memory comparison"
    "${python_bin}" -u "${project_root}/src/grpo_compare_runs.py" \
        --baseline-dir "${run_root}/baseline" \
        --ktr-dir "${run_root}/ktr" \
        --output-dir "${run_root}" \
        --require-pass \
        2>&1 | tee -a "${run_root}/terminal.log"
fi

if [[ "${variant_request}" == "both" ]]; then
    log "phase 7/7: PASS. Inspect ${run_root}/comparison.md, ktr/token_examples.md, and each variant training_summary.md"
elif [[ "${variant_request}" == "ktr" ]]; then
    log "phase 7/7: PASS. Inspect ${run_root}/ktr/token_examples.md and ${run_root}/ktr/training_summary.md"
else
    log "phase 7/7: PASS. Inspect ${run_root}/baseline/training_summary.md"
fi
