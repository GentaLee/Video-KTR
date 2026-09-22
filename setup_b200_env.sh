#!/usr/bin/env bash
# Provision the isolated Video-KTR runtime on a B200 shared-GPFS workspace.
#
# This script deliberately keeps the CUDA-coupled PyTorch and FlashAttention
# packages supplied by the B200 image.  It installs only the pinned Python
# stack from offline wheelhouses into a clean venv.  The CUDA-dependent image
# runtime is exposed only through explicit, ordered .pth entries.
# It is safe to run on the paired CPU pod (provisioning/import validation) or
# on the B200 GPU pod (which additionally proves the 8xB200/FA2 runtime).
#
# Required environment variables (no site-specific defaults are embedded):
#   B200_ROOT                 durable, shared GPFS root for this experiment
#   B200_REPO_ROOT            Video-KTR source checkout below B200_ROOT
#   B200_WHEELHOUSE_COMPAT    staged Transformers/tokenizers/AV wheelhouse
#   B200_WHEELHOUSE_GRPO      staged TRL/DeepSpeed wheelhouse
#
# Optional:
#   B200_MODEL_PATH            local checkpoint to verify after setup
#   B200_MODEL_INTEGRITY_MANIFEST defaults to the pinned repository manifest
#   B200_MODEL_INTEGRITY_REPORT durable non-authoritative verified-copy output
#   B200_BASE_PYTHON          defaults to /opt/venv/bin/python
#   B200_ENV_MANIFEST         defaults to $B200_ROOT/environment-manifest.json
#   B200_ENV_GPU_RUNTIME_CHECK defaults to 1; set to 0 only when provisioning
#                              on an occupied GPU node and defer the FA2
#                              kernel exercise to somke-b200.sh.
#   B200_RECREATE_VENV         defaults to 0; set to 1 to move aside a legacy
#                              venv created with system-site-packages.

set -Eeuo pipefail

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] [b200-env] %s\n' -1 "$*"
}

die() {
    log "ERROR: $*"
    exit 2
}

trap 'rc=$?; log "FAILED (exit=${rc}, line=${LINENO})"; exit "${rc}"' ERR

require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "required environment variable is unset: ${name}"
}

for required in B200_ROOT B200_REPO_ROOT B200_WHEELHOUSE_COMPAT B200_WHEELHOUSE_GRPO; do
    require_env "${required}"
done

[[ "${B200_ROOT}" == /* && "${B200_ROOT}" != "/" ]] || die "B200_ROOT must be a non-root absolute path"
[[ "${B200_REPO_ROOT}" == /* ]] || die "B200_REPO_ROOT must be an absolute path"
[[ "${B200_WHEELHOUSE_COMPAT}" == /* ]] || die "B200_WHEELHOUSE_COMPAT must be an absolute path"
[[ "${B200_WHEELHOUSE_GRPO}" == /* ]] || die "B200_WHEELHOUSE_GRPO must be an absolute path"

[[ -d "${B200_ROOT}" ]] || die "B200_ROOT does not exist: ${B200_ROOT}"
[[ -d "${B200_REPO_ROOT}" ]] || die "B200_REPO_ROOT does not exist: ${B200_REPO_ROOT}"
[[ -d "${B200_WHEELHOUSE_COMPAT}" ]] || die "B200_WHEELHOUSE_COMPAT does not exist: ${B200_WHEELHOUSE_COMPAT}"
[[ -d "${B200_WHEELHOUSE_GRPO}" ]] || die "B200_WHEELHOUSE_GRPO does not exist: ${B200_WHEELHOUSE_GRPO}"

b200_root="$(realpath -e "${B200_ROOT}")"
repo_root="$(realpath -e "${B200_REPO_ROOT}")"
compat_wheelhouse="$(realpath -e "${B200_WHEELHOUSE_COMPAT}")"
grpo_wheelhouse="$(realpath -e "${B200_WHEELHOUSE_GRPO}")"
case "${repo_root}/" in
    "${b200_root}/"*) ;;
    *) die "B200_REPO_ROOT must be below B200_ROOT (got ${repo_root}, root ${b200_root})" ;;
esac

for source_dir in \
    "${repo_root}/src/qwen-vl-utils/src" \
    "${repo_root}/src/r1-v/src" \
    "${repo_root}/src"; do
    [[ -d "${source_dir}" ]] || die "vendored source directory is missing: ${source_dir}"
done

# A venv on overlay/local scratch is unsafe: the paired CPU and GPU pods would
# see different packages and an image restart could erase the environment.
mount_type="$(findmnt -T "${b200_root}" -no FSTYPE 2>/dev/null | tr -d '[:space:]' || true)"
[[ "${mount_type}" == "gpfs" ]] || die "B200_ROOT must resolve to shared GPFS, got filesystem type '${mount_type:-unknown}'"

base_python="${B200_BASE_PYTHON:-/opt/venv/bin/python}"
[[ -x "${base_python}" ]] || die "B200 base Python is unavailable: ${base_python}"
gpu_runtime_check="${B200_ENV_GPU_RUNTIME_CHECK:-1}"
[[ "${gpu_runtime_check}" == "0" || "${gpu_runtime_check}" == "1" ]] \
    || die "B200_ENV_GPU_RUNTIME_CHECK must be 0 or 1, got '${gpu_runtime_check}'"

venv_dir="${b200_root}/venv"
venv_python="${venv_dir}/bin/python"
manifest_path="${B200_ENV_MANIFEST:-${b200_root}/environment-manifest.json}"
[[ "${manifest_path}" == /* ]] || die "B200_ENV_MANIFEST must be an absolute path"
manifest_path="$(realpath -m "${manifest_path}")"
case "${manifest_path}" in
    "${b200_root}"/*) ;;
    *) die "B200_ENV_MANIFEST must be below B200_ROOT (got ${manifest_path})" ;;
esac
model_path="${B200_MODEL_PATH:-}"
model_integrity_manifest="${B200_MODEL_INTEGRITY_MANIFEST:-${repo_root}/manifests/video-r1-qwen25vl-7b-cot-sft-f71f0f1e22c015007fccd080eef87824fe292a10.json}"
model_integrity_report="${B200_MODEL_INTEGRITY_REPORT:-${b200_root}/model_integrity_verified.json}"
[[ "${model_integrity_manifest}" == /* ]] || die "B200_MODEL_INTEGRITY_MANIFEST must be an absolute path"
[[ "${model_integrity_report}" == /* ]] || die "B200_MODEL_INTEGRITY_REPORT must be an absolute path"
model_integrity_manifest="$(realpath -m "${model_integrity_manifest}")"
model_integrity_report="$(realpath -m "${model_integrity_report}")"
case "${model_integrity_report}" in
    "${b200_root}"/*) ;;
    *) die "B200_MODEL_INTEGRITY_REPORT must be below B200_ROOT (got ${model_integrity_report})" ;;
esac
if [[ -n "${model_path}" ]]; then
    [[ -d "${model_path}" ]] || die "B200_MODEL_PATH is not a directory: ${model_path}"
    model_path="$(realpath -e "${model_path}")"
    [[ -f "${model_integrity_manifest}" ]] || die "B200_MODEL_INTEGRITY_MANIFEST is unavailable: ${model_integrity_manifest}"
    case "${model_integrity_report}" in
        "${model_path}"|"${model_path}"/*)
            die "B200_MODEL_INTEGRITY_REPORT must be outside B200_MODEL_PATH so it cannot hash itself"
            ;;
    esac
fi

compat_files=(
    tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    av-14.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    huggingface_hub-0.34.4-py3-none-any.whl
    transformers-4.49.0.dev0-py3-none-any.whl
)
grpo_wheel_files=(
    accelerate-1.3.0-py3-none-any.whl
    datasets-3.2.0-py3-none-any.whl
    trl-0.16.0-py3-none-any.whl
    peft-0.14.0-py3-none-any.whl
    nltk-3.9.1-py3-none-any.whl
    einops-0.8.0-py3-none-any.whl
    pyarrow-18.1.0-cp312-cp312-manylinux_2_28_x86_64.whl
    pandas-2.2.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    dill-0.3.8-py3-none-any.whl
    multiprocess-0.70.16-py312-none-any.whl
    xxhash-3.5.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    hjson-3.1.0-py3-none-any.whl
    ninja-1.11.1.3-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
    nvidia_ml_py-12.560.30-py3-none-any.whl
    py_cpuinfo-9.0.0-py3-none-any.whl
    pydantic-2.10.5-py3-none-any.whl
    pydantic_core-2.27.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    annotated_types-0.7.0-py3-none-any.whl
    typing_extensions-4.12.2-py3-none-any.whl
    rich-13.9.4-py3-none-any.whl
    markdown_it_py-3.0.0-py3-none-any.whl
    mdurl-0.1.2-py3-none-any.whl
    absl_py-2.1.0-py3-none-any.whl
    sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    fsspec-2024.9.0-py3-none-any.whl
    aiohttp-3.11.11-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    aiohappyeyeballs-2.4.4-py3-none-any.whl
    aiosignal-1.3.2-py2.py3-none-any.whl
    async_timeout-5.0.1-py3-none-any.whl
    attrs-24.3.0-py3-none-any.whl
    frozenlist-1.5.0-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    multidict-6.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    propcache-0.2.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    yarl-1.18.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    python_dateutil-2.9.0.post0-py2.py3-none-any.whl
    pytz-2024.2-py2.py3-none-any.whl
    tzdata-2024.2-py2.py3-none-any.whl
    six-1.17.0-py2.py3-none-any.whl
    docstring_parser-0.16-py3-none-any.whl
    shtab-1.7.1-py3-none-any.whl
    colorama-0.4.6-py2.py3-none-any.whl
    regex-2024.11.6-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    joblib-1.4.2-py3-none-any.whl
    tyro-0.9.10-py3-none-any.whl
    deepspeed-0.15.4-py3-none-any.whl
    rouge_score-0.1.2-py3-none-any.whl
)

for filename in "${compat_files[@]}"; do
    [[ -f "${compat_wheelhouse}/${filename}" ]] || die "missing compatibility wheel: ${compat_wheelhouse}/${filename}"
done
for filename in "${grpo_wheel_files[@]}"; do
    [[ -f "${grpo_wheelhouse}/${filename}" ]] || die "missing GRPO artifact: ${grpo_wheelhouse}/${filename}"
done

gpu_mode=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    gpu_mode=1
    mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
    [[ "${#gpu_names[@]}" -eq 8 ]] || die "B200 GPU node must expose exactly 8 GPUs; found ${#gpu_names[@]}"
    for gpu_name in "${gpu_names[@]}"; do
        [[ "${gpu_name}" == *B200* ]] || die "expected only B200 GPUs; found '${gpu_name}'"
    done
    log "validated GPU node: 8xB200"
else
    log "no usable NVIDIA device found; provisioning/import validation will run in paired-CPU mode"
fi

if [[ -x "${venv_python}" ]] \
    && ! grep -Eq '^include-system-site-packages = false$' "${venv_dir}/pyvenv.cfg"; then
    if [[ "${B200_RECREATE_VENV:-0}" != "1" ]]; then
        die "existing venv is not isolated; set B200_RECREATE_VENV=1 to move it aside and rebuild"
    fi
    venv_resolved="$(realpath -e "${venv_dir}")"
    [[ "${venv_resolved}" == "${b200_root}/venv" ]] \
        || die "refusing to move an unexpected venv path: ${venv_resolved}"
    archived_venv="${b200_root}/venv.pre-isolation-$(date -u +%Y%m%dT%H%M%SZ)"
    log "moving legacy non-isolated venv aside to ${archived_venv}"
    mv "${venv_resolved}" "${archived_venv}"
fi

if [[ ! -x "${venv_python}" ]]; then
    log "creating isolated venv at ${venv_dir} from ${base_python}"
    "${base_python}" -m venv "${venv_dir}"
else
    [[ -f "${venv_dir}/pyvenv.cfg" ]] || die "existing venv is missing pyvenv.cfg: ${venv_dir}"
    grep -Eq '^include-system-site-packages = false$' "${venv_dir}/pyvenv.cfg" \
        || die "existing venv is not isolated: ${venv_dir}"
    log "reusing existing isolated venv: ${venv_dir}"
fi

venv_python_minor="$("${base_python}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
venv_site="${venv_dir}/lib/python${venv_python_minor}/site-packages"
[[ -d "${venv_site}" ]] || die "venv site-packages directory is unavailable: ${venv_site}"

# Repair the old early-sorting compatibility hook before invoking this venv
# again.  Do not derive this path with ``venv_python -S``: in Python 3.12,
# disabling ``site`` also suppresses the venv prefix adjustment and can point
# at the platform interpreter's site-packages instead.
legacy_distutils_pth_file="${venv_site}/video_ktr_distutils_compat.pth"
rm -f "${legacy_distutils_pth_file}"
"${venv_python}" -m pip --version >/dev/null 2>&1 \
    || die "pip is unavailable in ${venv_python}; do not install into the platform Python"

# A base interpreter can create a venv from a different Python prefix.  Add
# only the selected CUDA-ready platform's own site directories after the venv
# site directory, so pinned venv packages override the image while unrelated
# system packages (including a broken global Apex) cannot leak into training.
platform_prefix="$("${base_python}" - <<'PY'
import sys
print(sys.prefix)
PY
)"
mapfile -t platform_sites < <("${base_python}" - <<'PY'
from pathlib import Path
import site
import sys

prefix = Path(sys.prefix).resolve()
for raw_path in site.getsitepackages():
    path = Path(raw_path).resolve()
    try:
        path.relative_to(prefix)
    except ValueError:
        continue
    if path.is_dir():
        print(path)
PY
)
(( ${#platform_sites[@]} > 0 )) || die "B200 base Python has no usable site-packages directory below ${platform_prefix}"

# The selected image can expose its CUDA packages through a secondary prefix
# (for example, a platform .pth pointing at /usr/local) even though the base
# interpreter itself lives under /opt.  A blanket secondary site-packages
# path would also expose unrelated packages such as a non-NVIDIA ``apex``
# module, which Transformers mistakes for Apex AMP.  Build a tiny durable
# bridge containing only the CUDA-coupled modules and their metadata.
platform_bridge="${b200_root}/platform-runtime-bridge"
mkdir -p "${platform_bridge}"
mapfile -t platform_runtime_sources < <(PYTHONWARNINGS=ignore "${base_python}" - <<'PY'
from importlib import import_module, metadata
from pathlib import Path

specification = (
    ("torch", "torch"),
    ("torchgen", "torch"),
    ("functorch", "torch"),
    ("torchvision", "torchvision"),
    ("flash_attn", "flash-attn"),
    ("flash_attn_2_cuda", "flash-attn"),
    # FlashAttention's rotary path imports Triton at model-import time.  The
    # NGC B200 image supplies the CUDA-matched ``pytorch-triton`` build beside
    # FA2; expose it through the same narrow bridge rather than fetching an
    # arbitrary PyPI Triton wheel into the isolated venv.
    ("triton", "pytorch-triton"),
    ("triton_kernels", "triton_kernels"),
)
seen: set[Path] = set()
for module_name, distribution_name in specification:
    module = import_module(module_name)
    module_path = Path(module.__file__).resolve()
    source = module_path.parent if module_path.name == "__init__.py" else module_path
    distribution = Path(metadata.distribution(distribution_name)._path).resolve()
    for path in (source, distribution):
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        print(path)
PY
)
(( ${#platform_runtime_sources[@]} > 0 )) || die "could not discover the platform CUDA runtime modules"
for source_path in "${platform_runtime_sources[@]}"; do
    [[ "${source_path}" == /* && -e "${source_path}" ]] \
        || die "invalid platform CUDA runtime source: ${source_path}"
    bridge_entry="${platform_bridge}/$(basename "${source_path}")"
    if [[ -e "${bridge_entry}" || -L "${bridge_entry}" ]]; then
        [[ "$(realpath -e "${bridge_entry}")" == "$(realpath -e "${source_path}")" ]] \
            || die "platform runtime bridge entry points at an unexpected source: ${bridge_entry}"
    else
        ln -s "${source_path}" "${bridge_entry}"
    fi
done

# The source checkout deliberately wins over an arbitrary PyPI qwen-vl-utils
# release, while the pinned venv packages win over the B200 image's newer
# Transformers stack.  The .pth is durable and applies to torchrun workers.
pth_file="${venv_site}/video_ktr_vendored_sources.pth"
pth_tmp="${pth_file}.tmp.$$"
{
    printf '%s\n' "${repo_root}/src/qwen-vl-utils/src"
    printf '%s\n' "${repo_root}/src/r1-v/src"
    printf '%s\n' "${repo_root}/src"
} > "${pth_tmp}"
mv -f "${pth_tmp}" "${pth_file}"

platform_pth_file="${venv_site}/video_ktr_platform_runtime.pth"
platform_pth_tmp="${platform_pth_file}.tmp.$$"
printf '%s\n' "${platform_sites[@]}" "${platform_bridge}" > "${platform_pth_tmp}"
mv -f "${platform_pth_tmp}" "${platform_pth_file}"

# DeepSpeed 0.15.4 still imports ``distutils`` directly, while Python 3.12
# only supplies the compatible implementation after setuptools initializes
# its local shim.  A .pth import runs before torchrun/DeepSpeed imports in
# every worker, and the explicit mode avoids accidentally preferring a removed
# stdlib copy.
# ``.pth`` files are processed lexically.  This must sort *after* the
# platform-runtime entry above: the selected B200 image supplies setuptools,
# and importing it earlier emits a startup error before that path is visible.
# Remove the older name on rerun so an existing venv is repaired in place.
distutils_pth_file="${venv_site}/zz_video_ktr_distutils_compat.pth"
distutils_pth_tmp="${distutils_pth_file}.tmp.$$"
printf '%s\n' 'import os; os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local"); import setuptools' > "${distutils_pth_tmp}"
mv -f "${distutils_pth_tmp}" "${distutils_pth_file}"
rm -f "${legacy_distutils_pth_file}"

mkdir -p "${b200_root}/.pip-cache" "${b200_root}/.pip-tmp" "$(dirname "${manifest_path}")"
export PIP_CACHE_DIR="${b200_root}/.pip-cache"
export TMPDIR="${b200_root}/.pip-tmp"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INDEX=1
export PYTHONNOUSERSITE=1
export SETUPTOOLS_USE_DISTUTILS=local
# Avoid an ambient control-plane PYTHONPATH shadowing the pinned venv during
# either package installation or the subsequent import gate.  The two durable
# .pth files above provide the only required project/platform paths.
export PYTHONPATH=""

log "installing pinned checkpoint-compatible packages from the offline compatibility wheelhouse"
compat_paths=()
for filename in "${compat_files[@]}"; do
    compat_paths+=("${compat_wheelhouse}/${filename}")
done
"${venv_python}" -m pip install --no-index --no-deps --upgrade --force-reinstall "${compat_paths[@]}"

log "installing the CUDA-independent GRPO Python stack from the offline wheelhouse"
grpo_paths=()
for filename in "${grpo_wheel_files[@]}"; do
    grpo_paths+=("${grpo_wheelhouse}/${filename}")
done
"${venv_python}" -m pip install --no-index --no-deps --upgrade --force-reinstall "${grpo_paths[@]}"

log "checking the resolved project dependency closure (excluding unrelated image packages)"
"${venv_python}" - <<'PY'
from collections import deque
from importlib import metadata

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

roots = (
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "av",
    "trl",
    "datasets",
    "accelerate",
    "deepspeed",
    "rouge_score",
    "peft",
    "einops",
    "nltk",
    "sentencepiece",
    "torch",
    "torchvision",
    "flash-attn",
    "pytorch-triton",
    "triton_kernels",
)
environment = default_environment()
environment["extra"] = ""
pending = deque(canonicalize_name(name) for name in roots)
seen: set[str] = set()
errors: list[str] = []

while pending:
    name = pending.popleft()
    if name in seen:
        continue
    seen.add(name)
    try:
        requirements = metadata.requires(name) or ()
    except metadata.PackageNotFoundError:
        errors.append(f"missing root/dependency distribution: {name}")
        continue
    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        dependency = canonicalize_name(requirement.name)
        try:
            installed = metadata.version(dependency)
        except metadata.PackageNotFoundError:
            errors.append(f"{name} requires {requirement}, but {dependency} is missing")
            continue
        if requirement.specifier and installed not in requirement.specifier:
            errors.append(f"{name} requires {requirement}, but installed {dependency}=={installed}")
        pending.append(dependency)

if errors:
    rendered = "\n".join(f"  - {item}" for item in sorted(set(errors)))
    raise SystemExit("project dependency closure is inconsistent:\n" + rendered)
print(f"[b200-env] project dependency closure verified ({len(seen)} distributions)")
PY

export B200_ENV_MANIFEST_PATH="${manifest_path}"
export B200_ENV_VENV_DIR="${venv_dir}"
export B200_ENV_REPO_ROOT="${repo_root}"
export B200_ENV_COMPAT_WHEELHOUSE="${compat_wheelhouse}"
export B200_ENV_GRPO_WHEELHOUSE="${grpo_wheelhouse}"
export B200_ENV_GPU_MODE="${gpu_mode}"
export B200_ENV_PLATFORM_PREFIX="${platform_prefix}"
export B200_ENV_PLATFORM_BRIDGE="${platform_bridge}"
export B200_ENV_GPU_RUNTIME_CHECK="${gpu_runtime_check}"

log "running import, pinned-version, vendored-source, and FlashAttention-2 gates"
"${venv_python}" - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import accelerate
import av
import datasets
import deepspeed
import flash_attn
import flash_attn_2_cuda
import huggingface_hub
import peft
import tokenizers
import torch
import torchvision
import transformers
import triton
import triton_kernels
import trl
import qwen_vl_utils
from transformers import Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.grpo import GRPOScriptArguments
from open_r1.trainer.video_ktr_grpo_trainer import VideoKTRGRPOTrainer

venv_dir = Path(os.environ["B200_ENV_VENV_DIR"]).resolve()
repo_root = Path(os.environ["B200_ENV_REPO_ROOT"]).resolve()
gpu_mode = os.environ["B200_ENV_GPU_MODE"] == "1"
gpu_runtime_check = os.environ["B200_ENV_GPU_RUNTIME_CHECK"] == "1"
manifest_path = Path(os.environ["B200_ENV_MANIFEST_PATH"])
platform_prefix = Path(os.environ["B200_ENV_PLATFORM_PREFIX"]).resolve()
platform_bridge = Path(os.environ["B200_ENV_PLATFORM_BRIDGE"]).resolve()

expected = {
    "transformers": "4.49.0.dev0",
    "tokenizers": "0.21.4",
    "av": "14.2.0",
    "trl": "0.16.0",
    "deepspeed": "0.15.4",
    "datasets": "3.2.0",
    "accelerate": "1.3.0",
    "peft": "0.14.0",
    "huggingface_hub": "0.34.4",
    "triton": "3.6.0",
    "pytorch_triton": "3.6.0+git9844da95.nv26.2",
    "triton_kernels": "1.0.0+git9844da95.nv26.2",
}
observed = {
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "av": av.__version__,
    "trl": trl.__version__,
    "deepspeed": deepspeed.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "peft": peft.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "triton": triton.__version__,
    "pytorch_triton": metadata.version("pytorch-triton"),
    "triton_kernels": metadata.version("triton_kernels"),
}
for name, version in expected.items():
    if observed[name] != version:
        raise SystemExit(f"version mismatch for {name}: {observed[name]} != {version}")

def starts_under(path, parent):
    return str(Path(path).resolve()).startswith(str(parent) + os.sep)

for package, module in (("transformers", transformers), ("tokenizers", tokenizers), ("av", av), ("huggingface_hub", huggingface_hub)):
    if not starts_under(module.__file__, venv_dir):
        raise SystemExit(f"{package} did not resolve from isolated venv: {module.__file__}")
if str(platform_bridge) not in sys.path:
    raise SystemExit(f"platform CUDA runtime bridge is absent from sys.path: {platform_bridge}")
if any(path.startswith("/usr/local/lib/python") and path.endswith("dist-packages") for path in sys.path):
    raise SystemExit("unexpected global /usr/local site-packages leaked into the isolated venv")
for package, module in (("torch", torch), ("torchvision", torchvision), ("flash_attn", flash_attn), ("triton", triton), ("triton_kernels", triton_kernels)):
    if starts_under(module.__file__, venv_dir):
        raise SystemExit(
            f"{package} must resolve from the selected B200 platform runtime bridge, not the venv: {module.__file__}"
        )

vendored_qwen = repo_root / "src" / "qwen-vl-utils" / "src"
vendored_r1 = repo_root / "src" / "r1-v" / "src"
if not starts_under(qwen_vl_utils.__file__, vendored_qwen):
    raise SystemExit(f"qwen_vl_utils did not resolve from vendored source: {qwen_vl_utils.__file__}")
import inspect
trainer_source = inspect.getsourcefile(VideoKTRGRPOTrainer)
if trainer_source is None or not starts_under(trainer_source, vendored_r1):
    raise SystemExit(f"VideoKTRGRPOTrainer did not resolve from vendored source: {trainer_source}")

extension = Path(flash_attn_2_cuda.__file__).resolve()
if not extension.is_file():
    raise SystemExit(f"FlashAttention-2 extension is unavailable: {extension}")

fa2_proofs = []
cuobjdump = shutil.which("cuobjdump")
if cuobjdump:
    result = subprocess.run([cuobjdump, "--list-elf", str(extension)], capture_output=True, text=True, check=False)
    inspection = result.stdout + result.stderr
    if "sm_100" in inspection.lower() or "sm100" in inspection.lower():
        fa2_proofs.append("cuobjdump:sm_100")
binary = extension.read_bytes()
if b"sm_100" in binary.lower() or b"sm100" in binary.lower() or b"compute_100" in binary.lower():
    fa2_proofs.append("extension-string:sm_100")

cuda = {
    "available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "devices": [],
}
if gpu_mode:
    if not torch.cuda.is_available():
        raise SystemExit("nvidia-smi found B200 GPUs but torch.cuda.is_available() is false")
    if torch.cuda.device_count() != 8:
        raise SystemExit(f"torch must expose 8 B200 GPUs, got {torch.cuda.device_count()}; check CUDA_VISIBLE_DEVICES")
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        if "B200" not in name or capability != (10, 0):
            raise SystemExit(f"GPU {index} is not expected B200 sm_100 hardware: {name}, capability={capability}")
        cuda["devices"].append({"index": index, "name": name, "capability": list(capability)})

    if gpu_runtime_check:
        # This intentionally tiny forward/backward proves that the inherited
        # FA2 extension can execute on sm_100, instead of merely being importable.
        from flash_attn import flash_attn_func
        q = torch.randn((1, 32, 2, 64), device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        out = flash_attn_func(q, k, v, causal=False)
        out.float().square().mean().backward()
        torch.cuda.synchronize(0)
        fa2_proofs.append("sm_100-runtime-forward-backward")
    else:
        fa2_proofs.append("sm_100-runtime-deferred-to-smoke")

if not fa2_proofs:
    raise SystemExit(
        "could not prove that the inherited FlashAttention-2 extension contains sm_100 support; "
        "run this on the B200 GPU pod or provide a platform FA2 build with embedded sm_100 metadata"
    )

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def wheelhouse_manifest(path):
    return {
        item.name: sha256_file(item)
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix in {".whl", ".gz"}
    }

git_revision = "unavailable"
try:
    git_revision = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
except Exception:
    pass

payload = {
    "schema": "video-ktr-b200-environment/v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "repo_root": str(repo_root),
    "repo_revision": git_revision,
    "python_executable": sys.executable,
    "python": {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "implementation": sys.implementation.name,
    },
    "venv": str(venv_dir),
    "platform_prefix": str(platform_prefix),
    "platform_bridge": str(platform_bridge),
    "versions": observed,
    "torch": {"version": torch.__version__, "cuda": torch.version.cuda, "file": torch.__file__},
    "torchvision": {"version": torchvision.__version__, "file": torchvision.__file__},
    "flash_attn": {
        "version": flash_attn.__version__,
        "file": flash_attn.__file__,
        "extension": str(extension),
        "sm_100_validation": fa2_proofs,
    },
    "triton": {
        "version": triton.__version__,
        "distribution": metadata.version("pytorch-triton"),
        "file": triton.__file__,
    },
    "triton_kernels": {
        "distribution": metadata.version("triton_kernels"),
        "file": triton_kernels.__file__,
    },
    "module_files": {
        "transformers": transformers.__file__,
        "tokenizers": tokenizers.__file__,
        "av": av.__file__,
        "huggingface_hub": huggingface_hub.__file__,
        "triton": triton.__file__,
        "triton_kernels": triton_kernels.__file__,
        "qwen_vl_utils": qwen_vl_utils.__file__,
    },
    "cuda": cuda,
    "gpu_runtime_check": gpu_runtime_check,
    "wheelhouses": {
        "compat": wheelhouse_manifest(Path(os.environ["B200_ENV_COMPAT_WHEELHOUSE"])),
        "grpo": wheelhouse_manifest(Path(os.environ["B200_ENV_GRPO_WHEELHOUSE"])),
    },
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
temporary = manifest_path.with_name(manifest_path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(manifest_path)

print("[b200-env] versions=" + repr(observed))
print("[b200-env] torch=" + torch.__version__ + " cuda=" + str(torch.version.cuda))
print("[b200-env] flash_attn=" + flash_attn.__version__ + " proofs=" + ",".join(fa2_proofs))
print("[b200-env] manifest=" + str(manifest_path))
PY

# The full launcher trusts the pinned repository manifest, not this local
# report.  Generating the report is an optional transfer-verification record;
# it is possible only after every model file exactly matches the pinned JSON.
# Environment provisioning still supports the paired CPU pod before checkpoint
# staging has completed.
if [[ -n "${model_path}" ]]; then
    log "verifying all top-level model files against pinned manifest and writing local report: ${model_integrity_report}"
    "${venv_python}" -u "${repo_root}/src/grpo_write_model_sha256_manifest.py" \
        --model-path "${model_path}" \
        --reference-manifest "${model_integrity_manifest}" \
        --output "${model_integrity_report}"
else
    log "B200_MODEL_PATH was not supplied; no local model-integrity report was written (run_full_b200.sh still uses its pinned repository manifest)"
fi

log "PASS: isolated B200 environment is ready; manifest=${manifest_path}"
