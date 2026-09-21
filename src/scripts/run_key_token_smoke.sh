#!/usr/bin/env bash
# One-command, offline, non-GRPO key-token attribution smoke/control test.
#
# Run this *on a GPU node* from any current directory.  It continuously prints
# phase progress and stores the same terminal log beside the report.

set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="${OUTPUT_ROOT:-${project_root}/artifacts/key-token-smoke}"
output_dir="${OUTPUT_DIR:-${output_root}/${timestamp}}"
gpu_index="${GPU_INDEX:-0}"
min_free_gib="${MIN_FREE_GIB:-24}"
python_bin="${PYTHON_BIN:-/usr/bin/python3.12}"
# Reuse the project-local runtime assembled by setup_grpo_env.sh.  SITE_PACKAGES
# remains available for an independently prepared compatible overlay.
site_dir="${SITE_PACKAGES:-${GRPO_SITE_PACKAGES:-${project_root}/.python-packages-grpo}}"
model_path="${MODEL_PATH:-}"
video_path="${VIDEO_PATH:-${project_root}/src/example_video/video1.mp4}"
question="${QUESTION:-Which move motion in the video lose the system energy?}"
case_id="${CASE_ID:-demo-video1}"
selection_mode="${SELECTION_MODE:-paper}"
if [[ -n "${SELECTION_SCOPE:-}" ]]; then
    selection_scope="${SELECTION_SCOPE}"
elif [[ "${selection_mode}" == "repo" ]]; then
    selection_scope="batch"
else
    selection_scope="per_completion"
fi
temporal_seed="${TEMPORAL_SEED:-}"
temporal_permutations="${TEMPORAL_PERMUTATIONS:-1}"
temporal_include_reverse="${TEMPORAL_INCLUDE_REVERSE:-0}"
verify_controls="${VERIFY_CONTROLS:-1}"
control_tolerance="${CONTROL_TOLERANCE:-1e-4}"
visual_perturbation="${VISUAL_PERTURBATION:-attention_mask_fixed_position}"
launcher_path="${LAUNCHER_PATH:-${project_root}/src/scripts/run_key_token_smoke.sh}"
run_config_sha256="${RUN_CONFIG_SHA256:-}"
entropy_ratio="${ENTROPY_RATIO:-0.2}"
visual_ratio="${VISUAL_RATIO:-0.2}"
temporal_ratio="${TEMPORAL_RATIO:-0.2}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-${gpu_index}}"
# The selector always uses logical cuda:0, which is the first physical device
# in CUDA_VISIBLE_DEVICES.  Check that same card before starting a run.
physical_gpu="${cuda_visible_devices%%,*}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] [launcher] %s\n' -1 "$*" | tee -a "${output_dir}/terminal.log"
}

on_error() {
    local exit_code=$?
    log "FAILED (exit=${exit_code}); inspect ${output_dir}/terminal.log"
    exit "${exit_code}"
}
trap on_error ERR

mkdir -p "${output_dir}"
log "phase 1/5: checking GPU, local model, and local dependency overlay"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi is unavailable; run this script on a CUDA GPU node."
    exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
    log "Python interpreter is unavailable: ${python_bin}"
    exit 2
fi
if [[ -z "${model_path}" ]]; then
    log "MODEL_PATH is required and must point at the local checkpoint directory."
    exit 2
fi
if [[ ! -d "${model_path}" ]]; then
    log "Model is unavailable: ${model_path}"
    exit 2
fi
if [[ ! -f "${video_path}" ]]; then
    log "Demo video is unavailable: ${video_path}"
    exit 2
fi
if [[ ! -d "${site_dir}" ]]; then
    log "Missing local Python overlay: ${site_dir}. Run ./setup_grpo_env.sh on this GPU node first, or set SITE_PACKAGES."
    exit 2
fi

if [[ ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
    log "CUDA_VISIBLE_DEVICES must begin with a numeric GPU index for the launcher memory check; got: ${cuda_visible_devices}"
    exit 2
fi
free_mib="$(nvidia-smi --id="${physical_gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
    log "Could not determine free memory for physical GPU ${physical_gpu}: ${free_mib}"
    exit 2
fi
required_mib=$((min_free_gib * 1024))
log "physical GPU ${physical_gpu}: ${free_mib} MiB free; required minimum: ${required_mib} MiB"
if (( free_mib < required_mib )); then
    log "Not enough free memory; leaving existing GPU jobs untouched. Set GPU_INDEX/MIN_FREE_GIB only if you know capacity is available."
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${site_dir}:${project_root}/src/qwen-vl-utils/src:${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

log "phase 2/5: verifying local imports (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
"${python_bin}" - <<'PY' 2>&1 | tee -a "${output_dir}/terminal.log"
import av
import torch
import transformers
from qwen_vl_utils import process_vision_info

assert torch.cuda.is_available(), "CUDA is not available to this Python process"
print(
    "[launcher-preflight] "
    f"torch={torch.__version__}; cuda={torch.version.cuda}; "
    f"transformers={transformers.__version__}; av={av.__version__}; "
    f"device={torch.cuda.get_device_name(0)}",
    flush=True,
)
PY

log "phase 3/5: starting generation, position-preserving visual ablation, and temporal probes; heartbeat lines follow every 10s"
log "phase 4/5: artifacts will be written under ${output_dir}"
set -o pipefail
selector_args=(
    --model-path "${model_path}" \
    --video-path "${video_path}" \
    --question "${question}" \
    --case-id "${case_id}" \
    --output-dir "${output_dir}" \
    --nframes "${NFRAMES:-4}" \
    --max-pixels "${MAX_PIXELS:-100352}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
    --min-new-tokens "${MIN_NEW_TOKENS:-96}" \
    --temperature "${TEMPERATURE:-0.7}" \
    --top-p "${TOP_P:-0.95}" \
    --seed "${SEED:-42}" \
    --gpu-memory-fraction "${GPU_MEMORY_FRACTION:-0.18}" \
    --physical-gpu-id "${physical_gpu}" \
    --selection-mode "${selection_mode}" \
    --selection-scope "${selection_scope}" \
    --entropy-ratio "${entropy_ratio}" \
    --visual-ratio "${visual_ratio}" \
    --temporal-ratio "${temporal_ratio}" \
    --temporal-permutations "${temporal_permutations}" \
    --visual-perturbation "${visual_perturbation}" \
    --control-tolerance "${control_tolerance}" \
    --model-revision "${MODEL_REVISION:-f71f0f1e22c015007fccd080eef87824fe292a10}" \
    --transformers-source-commit "${TRANSFORMERS_SOURCE_COMMIT:-336dc69d63d56f232a183a3e7f52790429b871ef}" \
    --launcher-path "${launcher_path}" \
    --heartbeat-seconds "${HEARTBEAT_SECONDS:-10}" \
)
if [[ -n "${temporal_seed}" ]]; then
    selector_args+=(--temporal-seed "${temporal_seed}")
fi
if [[ -n "${run_config_sha256}" ]]; then
    selector_args+=(--run-config-sha256 "${run_config_sha256}")
fi
if [[ "${temporal_include_reverse}" == "1" ]]; then
    selector_args+=(--temporal-include-reverse)
fi
if [[ "${verify_controls}" == "1" ]]; then
    selector_args+=(--verify-controls)
fi
"${python_bin}" -u "${project_root}/src/key_token_selector.py" "${selector_args[@]}" \
    2>&1 | tee -a "${output_dir}/terminal.log"

log "phase 5/5: success. Open ${output_dir}/report.md, ${output_dir}/result.json, and ${output_dir}/status.json"
