#!/usr/bin/env bash
# Sequential, resumable non-GRPO validation launcher.
#
# This is intentionally a launcher only: run it manually on a GPU node.  It
# does not download data, alter model weights, or start distributed training.
# Each case is isolated in its own output directory so an interrupted batch can
# resume safely with RESUME=1.

set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
manifest="${MANIFEST:-${project_root}/src/validation_manifests/demo_p3.tsv}"
video_root="${VIDEO_ROOT:-${project_root}}"
run_dir="${RUN_DIR:-${project_root}/artifacts/key-token-validation/${timestamp}}"
python_bin="${PYTHON_BIN:-/usr/bin/python3.12}"
generation_seed_text="${SEEDS:-42,43,44,45,46,47,48,49}"
repeat_seed_text="${REPEAT_SEEDS:-42}"
resume="${RESUME:-1}"
continue_on_error="${CONTINUE_ON_ERROR:-0}"
site_dir="${SITE_PACKAGES:-${GRPO_SITE_PACKAGES:-${project_root}/.python-packages-grpo}}"
model_path="${MODEL_PATH:-}"
model_revision="${MODEL_REVISION:-f71f0f1e22c015007fccd080eef87824fe292a10}"
transformers_source_commit="${TRANSFORMERS_SOURCE_COMMIT:-336dc69d63d56f232a183a3e7f52790429b871ef}"
selection_mode="${SELECTION_MODE:-paper}"
if [[ -n "${SELECTION_SCOPE:-}" ]]; then
    selection_scope="${SELECTION_SCOPE}"
elif [[ "${selection_mode}" == "repo" ]]; then
    selection_scope="batch"
else
    selection_scope="per_completion"
fi
entropy_ratio="${ENTROPY_RATIO:-0.2}"
visual_ratio="${VISUAL_RATIO:-0.2}"
temporal_ratio="${TEMPORAL_RATIO:-0.2}"
temperature="${TEMPERATURE:-0.7}"
top_p="${TOP_P:-0.95}"
temporal_permutations="${TEMPORAL_PERMUTATIONS:-5}"
temporal_include_reverse="${TEMPORAL_INCLUDE_REVERSE:-1}"
verify_controls="${VERIFY_CONTROLS:-1}"
control_tolerance="${CONTROL_TOLERANCE:-1e-4}"
visual_perturbation="${VISUAL_PERTURBATION:-attention_mask_fixed_position}"
max_new_tokens="${MAX_NEW_TOKENS:-512}"
min_new_tokens="${MIN_NEW_TOKENS:-96}"
nframes="${NFRAMES:-8}"
max_pixels="${MAX_PIXELS:-100352}"
gpu_index="${GPU_INDEX:-0}"
min_free_gib="${MIN_FREE_GIB:-24}"
gpu_memory_fraction="${GPU_MEMORY_FRACTION:-0.18}"
# Batch mode treats GPU_INDEX as the physical card unambiguously.  It passes
# this through as CUDA_VISIBLE_DEVICES instead of inheriting a stale shell
# mapping, while the selector itself uses logical cuda:0.
cuda_visible_devices="${gpu_index}"
validation_launcher_path="${project_root}/src/scripts/run_key_token_validation.sh"
smoke_launcher_path="${project_root}/src/scripts/run_key_token_smoke.sh"
selector_path="${project_root}/src/key_token_selector.py"
attribution_path="${project_root}/src/key_token_attribution.py"
summary_path="${project_root}/src/key_token_validation_summary.py"

if [[ "${manifest}" != /* ]]; then
    manifest="${project_root}/${manifest}"
fi
if [[ "${video_root}" != /* ]]; then
    video_root="${project_root}/${video_root}"
fi
if [[ "${run_dir}" != /* ]]; then
    run_dir="${project_root}/${run_dir}"
fi

mkdir -p "${run_dir}"
validation_log="${run_dir}/terminal.log"
failures_tsv="${run_dir}/failures.tsv"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] [validation] %s\n' -1 "$*" | tee -a "${validation_log}"
}

on_error() {
    local exit_code=$?
    log "FAILED (exit=${exit_code}); artifacts and progress remain in ${run_dir}"
    exit "${exit_code}"
}
trap on_error ERR

resolve_video() {
    local raw_path="$1"
    if [[ "${raw_path}" == /* && -f "${raw_path}" ]]; then
        printf '%s\n' "${raw_path}"
    elif [[ -f "${project_root}/${raw_path}" ]]; then
        printf '%s\n' "${project_root}/${raw_path}"
    elif [[ -f "${video_root}/${raw_path}" ]]; then
        printf '%s\n' "${video_root}/${raw_path}"
    else
        return 1
    fi
}

completed_case() {
    local case_dir="$1"
    local expected_case_id="$2"
    local expected_seed="$3"
    local expected_config_sha256="$4"
    "${python_bin}" - "${case_dir}/status.json" "${case_dir}/result.json" "${expected_case_id}" "${expected_seed}" "${expected_config_sha256}" <<'PY' >/dev/null 2>&1
import hashlib
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
expected_case_id = sys.argv[3]
expected_seed = int(sys.argv[4])
expected_config_sha256 = sys.argv[5]
if not status_path.is_file() or not result_path.is_file():
    raise SystemExit(1)
status = json.loads(status_path.read_text(encoding="utf-8"))
raw = result_path.read_bytes()
result = json.loads(raw)
if status.get("status") != "success":
    raise SystemExit(1)
if status.get("result_sha256") != hashlib.sha256(raw).hexdigest():
    raise SystemExit(1)
metadata = result.get("metadata", {})
if metadata.get("case_id") != expected_case_id or metadata.get("seed") != expected_seed:
    raise SystemExit(1)
if status.get("run_config_sha256") != expected_config_sha256:
    raise SystemExit(1)
if metadata.get("run_config_sha256") != expected_config_sha256:
    raise SystemExit(1)
PY
}

file_sha256() {
    sha256sum -- "$1" | awk '{print $1}'
}

run_config_digest() {
    local case_id="$1"
    local video_path="$2"
    local video_sha256="$3"
    local question="$4"
    local seed="$5"
    local temporal_seed="$6"
    {
        printf '%s\0' \
            'schema=key-token-validation-v1' \
            "batch_contract_sha256=${batch_contract_sha256}" \
            "manifest_sha256=${manifest_sha256}" \
            "case_id=${case_id}" \
            "video_path=${video_path}" \
            "video_sha256=${video_sha256}" \
            "question=${question}" \
            "generation_seed=${seed}" \
            "temporal_seed=${temporal_seed}" \
            "model_path=${model_path}" \
            "model_revision=${model_revision}" \
            "transformers_source_commit=${transformers_source_commit}" \
            "selection_mode=${selection_mode}" \
            "selection_scope=${selection_scope}" \
            "entropy_ratio=${entropy_ratio}" \
            "visual_ratio=${visual_ratio}" \
            "temporal_ratio=${temporal_ratio}" \
            "temperature=${temperature}" \
            "top_p=${top_p}" \
            "temporal_permutations=${temporal_permutations}" \
            "temporal_include_reverse=${temporal_include_reverse}" \
            "verify_controls=${verify_controls}" \
            "control_tolerance=${control_tolerance}" \
            "visual_perturbation=${visual_perturbation}" \
            "max_new_tokens=${max_new_tokens}" \
            "min_new_tokens=${min_new_tokens}" \
            "nframes=${nframes}" \
            "max_pixels=${max_pixels}" \
            "gpu_memory_fraction=${gpu_memory_fraction}" \
            "site_packages=${site_dir}" \
            "python_bin=${python_bin}" \
            "validation_launcher_sha256=${validation_launcher_sha256}" \
            "smoke_launcher_sha256=${smoke_launcher_sha256}" \
            "selector_sha256=${selector_sha256}" \
            "attribution_sha256=${attribution_sha256}" \
            "summary_sha256=${summary_sha256}"
    } | sha256sum | awk '{print $1}'
}

batch_contract_digest() {
    {
        printf '%s\0' \
            'schema=key-token-validation-batch-v1' \
            "manifest_sha256=${manifest_sha256}" \
            "model_path=${model_path}" \
            "model_revision=${model_revision}" \
            "transformers_source_commit=${transformers_source_commit}" \
            "selection_mode=${selection_mode}" \
            "selection_scope=${selection_scope}" \
            "entropy_ratio=${entropy_ratio}" \
            "visual_ratio=${visual_ratio}" \
            "temporal_ratio=${temporal_ratio}" \
            "temperature=${temperature}" \
            "top_p=${top_p}" \
            "temporal_permutations=${temporal_permutations}" \
            "temporal_include_reverse=${temporal_include_reverse}" \
            "verify_controls=${verify_controls}" \
            "control_tolerance=${control_tolerance}" \
            "visual_perturbation=${visual_perturbation}" \
            "max_new_tokens=${max_new_tokens}" \
            "min_new_tokens=${min_new_tokens}" \
            "nframes=${nframes}" \
            "max_pixels=${max_pixels}" \
            "gpu_memory_fraction=${gpu_memory_fraction}" \
            "gpu_index=${gpu_index}" \
            "cuda_visible_devices=${cuda_visible_devices}" \
            "site_packages=${site_dir}" \
            "python_bin=${python_bin}" \
            "validation_launcher_sha256=${validation_launcher_sha256}" \
            "smoke_launcher_sha256=${smoke_launcher_sha256}" \
            "selector_sha256=${selector_sha256}" \
            "attribution_sha256=${attribution_sha256}" \
            "summary_sha256=${summary_sha256}"
        for seed_index in "${!seeds[@]}"; do
            printf '%s\0' "seed[${seed_index}]=${seeds[seed_index]}"
        done
    } | sha256sum | awk '{print $1}'
}

if [[ ! -x "${python_bin}" ]]; then
    log "Python interpreter is unavailable: ${python_bin}"
    exit 2
fi
if [[ -z "${model_path}" ]]; then
    log "MODEL_PATH is required and must point at the local checkpoint directory."
    exit 2
fi
if [[ ! -d "${site_dir}" ]]; then
    log "Missing local Python overlay: ${site_dir}. Run ./setup_grpo_env.sh on this GPU node first, or set SITE_PACKAGES."
    exit 2
fi
if [[ ! -f "${manifest}" ]]; then
    log "Manifest is unavailable: ${manifest}"
    exit 2
fi
if [[ ! -d "${video_root}" ]]; then
    log "VIDEO_ROOT is unavailable: ${video_root}"
    exit 2
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    log "sha256sum is unavailable; it is required to make RESUME=1 configuration-safe."
    exit 2
fi
for required_source in "${validation_launcher_path}" "${smoke_launcher_path}" "${selector_path}" "${attribution_path}" "${summary_path}"; do
    if [[ ! -f "${required_source}" ]]; then
        log "Required validation source is unavailable: ${required_source}"
        exit 2
    fi
done
validation_launcher_sha256="$(file_sha256 "${validation_launcher_path}")"
smoke_launcher_sha256="$(file_sha256 "${smoke_launcher_path}")"
selector_sha256="$(file_sha256 "${selector_path}")"
attribution_sha256="$(file_sha256 "${attribution_path}")"
summary_sha256="$(file_sha256 "${summary_path}")"
manifest_sha256="$(file_sha256 "${manifest}")"

declare -a case_ids=()
declare -a case_videos=()
declare -a case_video_sha256s=()
declare -a case_questions=()
declare -A seen_case_ids=()
while IFS=$'\t' read -r case_id raw_video question extra || [[ -n "${case_id:-}" ]]; do
    [[ -z "${case_id:-}" || "${case_id:0:1}" == "#" ]] && continue
    if [[ -n "${extra:-}" || -z "${raw_video:-}" || -z "${question:-}" ]]; then
        log "Malformed manifest row for case '${case_id}'; expected exactly three tab-separated fields."
        exit 2
    fi
    if [[ ! "${case_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        log "Invalid case id '${case_id}'; use only letters, digits, '.', '_' and '-'."
        exit 2
    fi
    if [[ -n "${seen_case_ids[${case_id}]:-}" ]]; then
        log "Duplicate case id in manifest: ${case_id}"
        exit 2
    fi
    if ! resolved_video="$(resolve_video "${raw_video}")"; then
        log "Video for case '${case_id}' was not found: ${raw_video} (project=${project_root}, VIDEO_ROOT=${video_root})"
        exit 2
    fi
    if ! resolved_video_sha256="$(file_sha256 "${resolved_video}")"; then
        log "Could not hash video for case '${case_id}': ${resolved_video}"
        exit 2
    fi
    seen_case_ids["${case_id}"]=1
    case_ids+=("${case_id}")
    case_videos+=("${resolved_video}")
    case_video_sha256s+=("${resolved_video_sha256}")
    case_questions+=("${question}")
done < "${manifest}"

if (( ${#case_ids[@]} == 0 )); then
    log "Manifest contains no runnable video case: ${manifest}"
    exit 2
fi

seed_words="${generation_seed_text//,/ } ${repeat_seed_text//,/ }"
read -r -a seeds <<< "${seed_words}"
if (( ${#seeds[@]} == 0 )); then
    log "SEEDS/REPEAT_SEEDS produced no seed."
    exit 2
fi
for seed in "${seeds[@]}"; do
    if [[ ! "${seed}" =~ ^-?[0-9]+$ ]]; then
        log "Invalid seed: ${seed}"
        exit 2
    fi
done

batch_contract_sha256="$(batch_contract_digest)"
batch_contract_path="${run_dir}/run_contract.sha256"
if [[ -e "${batch_contract_path}" ]]; then
    existing_batch_contract="$(tr -d '\r\n' < "${batch_contract_path}")"
    if [[ "${existing_batch_contract}" != "${batch_contract_sha256}" ]]; then
        log "RUN_DIR already belongs to a different immutable validation contract; preserving it. Use a new RUN_DIR."
        exit 2
    fi
elif [[ -d "${run_dir}/cases" ]] && existing_case_artifact="$(find "${run_dir}/cases" -type f \( -name status.json -o -name result.json -o -name result.jsonl -o -name report.md \) -print -quit)" && [[ -n "${existing_case_artifact}" ]]; then
    log "RUN_DIR contains prior case artifacts but no immutable contract; preserving them. Use a new RUN_DIR."
    exit 2
else
    printf '%s\n' "${batch_contract_sha256}" > "${batch_contract_path}"
fi

# Never overwrite completed evidence.  The digest below includes the full
# manifest, video content, CLI-equivalent settings, and source hashes, so a
# matching marker is the only safe thing to resume inside this RUN_DIR.
for case_index in "${!case_ids[@]}"; do
    case_id="${case_ids[case_index]}"
    video_path="${case_videos[case_index]}"
    video_sha256="${case_video_sha256s[case_index]}"
    question="${case_questions[case_index]}"
    for seed_index in "${!seeds[@]}"; do
        seed="${seeds[seed_index]}"
        run_index=$((case_index * ${#seeds[@]} + seed_index + 1))
        case_dir="${run_dir}/cases/${case_id}/run-$(printf '%03d' "${run_index}")-seed-${seed}"
        temporal_seed=$((seed + 1000003))
        run_config_sha256="$(run_config_digest "${case_id}" "${video_path}" "${video_sha256}" "${question}" "${seed}" "${temporal_seed}")"
        if [[ -e "${case_dir}/status.json" || -e "${case_dir}/result.json" || -e "${case_dir}/result.jsonl" || -e "${case_dir}/report.md" ]]; then
            if completed_case "${case_dir}" "${case_id}" "${seed}" "${run_config_sha256}"; then
                if [[ "${resume}" != "1" ]]; then
                    log "Completed case=${case_id} seed=${seed} already exists; use a new RUN_DIR instead of overwriting it."
                    exit 2
                fi
            else
                log "Existing artifacts for case=${case_id} seed=${seed} do not match this run configuration; preserving them. Use a new RUN_DIR for changed code/data/settings."
                exit 2
            fi
        fi
    done
done

expected_runs=$(( ${#case_ids[@]} * ${#seeds[@]} ))
cp -- "${manifest}" "${run_dir}/manifest.input.tsv"
{
    printf '# case_id\tresolved_video_path\tvideo_sha256\tquestion\n'
    for index in "${!case_ids[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "${case_ids[index]}" "${case_videos[index]}" "${case_video_sha256s[index]}" "${case_questions[index]}"
    done
} > "${run_dir}/manifest.lock.tsv"
printf '# run_index\tcase_id\tseed\texit_code\tdirectory\n' > "${failures_tsv}"

log "prepared ${#case_ids[@]} case(s) × ${#seeds[@]} sequential seed(s) = ${expected_runs} run(s)"
log "run_dir=${run_dir}; RESUME=${resume}; controls=${verify_controls}; temporal probes=${temporal_permutations}; physical_gpu=${gpu_index}; contract=${batch_contract_sha256:0:12}"

run_index=0
failure_count=0
for case_index in "${!case_ids[@]}"; do
    case_id="${case_ids[case_index]}"
    video_path="${case_videos[case_index]}"
    video_sha256="${case_video_sha256s[case_index]}"
    question="${case_questions[case_index]}"
    for seed in "${seeds[@]}"; do
        run_index=$((run_index + 1))
        case_dir="${run_dir}/cases/${case_id}/run-$(printf '%03d' "${run_index}")-seed-${seed}"
        temporal_seed=$((seed + 1000003))
        run_config_sha256="$(run_config_digest "${case_id}" "${video_path}" "${video_sha256}" "${question}" "${seed}" "${temporal_seed}")"
        if completed_case "${case_dir}" "${case_id}" "${seed}" "${run_config_sha256}"; then
            log "[${run_index}/${expected_runs}] case=${case_id} seed=${seed} SKIP (verified success marker)"
            continue
        fi
        mkdir -p "${case_dir}"
        log "[${run_index}/${expected_runs}] case=${case_id} seed=${seed} temporal_seed=${temporal_seed} config=${run_config_sha256:0:12} START"
        if env \
            OUTPUT_DIR="${case_dir}" \
            OUTPUT_ROOT="${run_dir}" \
            CASE_ID="${case_id}" \
            VIDEO_PATH="${video_path}" \
            QUESTION="${question}" \
            SEED="${seed}" \
            TEMPORAL_SEED="${temporal_seed}" \
            TEMPORAL_PERMUTATIONS="${temporal_permutations}" \
            TEMPORAL_INCLUDE_REVERSE="${temporal_include_reverse}" \
            VERIFY_CONTROLS="${verify_controls}" \
            CONTROL_TOLERANCE="${control_tolerance}" \
            VISUAL_PERTURBATION="${visual_perturbation}" \
            MAX_NEW_TOKENS="${max_new_tokens}" \
            MIN_NEW_TOKENS="${min_new_tokens}" \
            NFRAMES="${nframes}" \
            MAX_PIXELS="${max_pixels}" \
            GPU_INDEX="${gpu_index}" \
            CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
            MIN_FREE_GIB="${min_free_gib}" \
            GPU_MEMORY_FRACTION="${gpu_memory_fraction}" \
            SITE_PACKAGES="${site_dir}" \
            GRPO_SITE_PACKAGES="${site_dir}" \
            MODEL_PATH="${model_path}" \
            MODEL_REVISION="${model_revision}" \
            TRANSFORMERS_SOURCE_COMMIT="${transformers_source_commit}" \
            SELECTION_MODE="${selection_mode}" \
            SELECTION_SCOPE="${selection_scope}" \
            ENTROPY_RATIO="${entropy_ratio}" \
            VISUAL_RATIO="${visual_ratio}" \
            TEMPORAL_RATIO="${temporal_ratio}" \
            TEMPERATURE="${temperature}" \
            TOP_P="${top_p}" \
            RUN_CONFIG_SHA256="${run_config_sha256}" \
            PYTHON_BIN="${python_bin}" \
            LAUNCHER_PATH="${validation_launcher_path}" \
            "${smoke_launcher_path}" 2>&1 | tee -a "${validation_log}"; then
            child_status=0
        else
            pipeline_statuses=("${PIPESTATUS[@]}")
            child_status="${pipeline_statuses[0]:-1}"
            # A logging failure also makes the run unusable even if the child
            # happened to finish, because the promised live/auditable log is
            # then incomplete.
            if (( child_status == 0 )); then
                child_status="${pipeline_statuses[1]:-1}"
            fi
        fi
        if (( child_status == 0 )); then
            log "[${run_index}/${expected_runs}] case=${case_id} seed=${seed} PASS"
        else
            failure_count=$((failure_count + 1))
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "${run_index}" "${case_id}" "${seed}" "${child_status}" "${case_dir}" >> "${failures_tsv}"
            log "[${run_index}/${expected_runs}] case=${case_id} seed=${seed} FAIL (exit=${child_status})"
            if [[ "${continue_on_error}" != "1" ]]; then
                log "Stopping after first failure; rerun with RESUME=1 after fixing it, or CONTINUE_ON_ERROR=1 to collect independent failures."
                break 2
            fi
        fi
    done
done

log "summarizing completed artifacts"
if "${python_bin}" "${project_root}/src/key_token_validation_summary.py" \
    --run-dir "${run_dir}" \
    --expected-runs "${expected_runs}" \
    --require-complete 2>&1 | tee -a "${validation_log}"; then
    summary_status=0
else
    pipeline_statuses=("${PIPESTATUS[@]}")
    summary_status="${pipeline_statuses[0]:-1}"
    if (( summary_status == 0 )); then
        summary_status="${pipeline_statuses[1]:-1}"
    fi
fi

if (( failure_count > 0 || summary_status != 0 )); then
    log "Validation ended with incomplete/failed cases; inspect ${run_dir}/summary.md and ${failures_tsv}."
    exit 1
fi
log "Validation PASS. Open ${run_dir}/summary.md and per-case reports under ${run_dir}/cases/."
