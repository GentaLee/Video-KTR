#!/usr/bin/env bash
# Build the isolated, offline runtime used by Video-KTR direct GRPO launchers.
# Run this on the GPU node only.  It never modifies the system Python.

set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The GPU image's Python was built without ensurepip, so a conventional venv
# cannot be created there.  Use a project-local site overlay instead: it keeps
# CUDA-coupled packages (torch/torchvision/flash-attn) from the GPU image while
# placing the Python-only GRPO stack ahead of its incompatible global packages.
site_dir="${GRPO_SITE_PACKAGES:-${project_root}/.python-packages-grpo}"
wheelhouse="${GRPO_WHEELHOUSE:-${project_root}/.wheelhouse-grpo}"
# The checkpoint-compatible Transformers wheel and its tokenizer/video
# dependencies are staged separately from the GRPO runtime wheelhouse.  Keep
# this location configurable too: neither wheelhouse is committed to Git.
compat_wheelhouse="${COMPAT_WHEELHOUSE:-${project_root}/.wheelhouse}"
base_python="${GRPO_BASE_PYTHON:-/usr/bin/python3.12}"
cache_dir="${project_root}/.pip-cache-grpo"
temp_dir="${project_root}/.tmp-pip-grpo"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] [grpo-env] %s\n' -1 "$*"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: this setup must run on a CUDA GPU node (nvidia-smi is unavailable)"
    exit 2
fi
if [[ ! -x "${base_python}" ]]; then
    log "ERROR: base Python is unavailable: ${base_python}"
    exit 2
fi
if [[ ! -d "${wheelhouse}" ]]; then
    log "ERROR: offline wheelhouse is unavailable: ${wheelhouse}"
    exit 2
fi
if [[ ! -d "${compat_wheelhouse}" ]]; then
    log "ERROR: checkpoint-compatible wheelhouse is unavailable: ${compat_wheelhouse}"
    exit 2
fi
for wheel in \
    tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    av-14.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    transformers-4.49.0.dev0-py3-none-any.whl; do
    if [[ ! -f "${compat_wheelhouse}/${wheel}" ]]; then
        log "ERROR: required checkpoint-compatible wheel is unavailable: ${compat_wheelhouse}/${wheel}"
        exit 2
    fi
done

mkdir -p "${cache_dir}" "${temp_dir}"
export PIP_CACHE_DIR="${cache_dir}"
export TMPDIR="${temp_dir}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
mkdir -p "${site_dir}"
python_bin="${base_python}"
pip_cmd=("${base_python}" -m pip)

if ! "${pip_cmd[@]}" --version >/dev/null 2>&1; then
    log "ERROR: pip is unavailable for ${base_python}"
    exit 2
fi

# The staged wheelhouses intentionally contain the CUDA-independent GRPO
# layer and the checkpoint-compatible Transformers artifacts.  They do not
# yet include every transitive pure-Python dependency of Transformers,
# Accelerate, and the image processor.  Verify the GPU image supplies that
# explicit base contract before writing a partial overlay; a clean CPython is
# not a supported reconstruction target.  A complete auxiliary wheel bundle
# alone is not enough with this version of the script: it also needs a locked
# full-overlay install path.  Until that extension is implemented and tested,
# use the validated GPU image rather than bypassing this gate.
log "checking explicit GPU-image prerequisites required by the partial offline wheelhouses"
env -u PYTHONPATH PYTHONNOUSERSITE=1 "${base_python}" - <<'PY'
import importlib

requirements = {
    "torch": "PyTorch supplied by the GPU image",
    "torchvision": "torchvision supplied by the GPU image",
    "flash_attn": "FlashAttention supplied by the GPU image",
    "numpy": "Transformers/image processing dependency",
    "PIL": "Pillow image-processing dependency",
    "huggingface_hub": "Transformers/Accelerate dependency",
    "filelock": "Transformers dependency",
    "packaging": "Transformers/Accelerate dependency",
    "yaml": "PyYAML dependency",
    "requests": "Transformers/Datasets dependency",
    "safetensors": "Transformers/Accelerate dependency",
    "tqdm": "Transformers/Datasets dependency",
    "psutil": "Accelerate dependency",
    "msgpack": "DeepSpeed dependency",
}
missing = []
for module, purpose in requirements.items():
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module} ({purpose}; {type(exc).__name__}: {exc})")
if missing:
    raise SystemExit(
        "GPU-image prerequisite gate failed. The staged wheelhouses are partial; "
        "do not attempt an online install on this node. Missing:\\n- "
        + "\\n- ".join(missing)
        + "\\nUse the validated GPU image. Clean-Python reconstruction is unsupported by this"
        + " version of setup_grpo_env.sh; it needs both a complete, version-locked wheel bundle"
        + " and a tested full-overlay install path before it can be enabled."
    )
PY

# This must be exported before building the two sdists below.  It lets their
# build processes see the already-installed pure-Python runtime dependencies
# without ever writing into the system site-packages directory.
export PYTHONPATH="${site_dir}:${project_root}/src:${project_root}/src/r1-v/src/open_r1:${project_root}/src/r1-v/src:${project_root}/src/qwen-vl-utils/src${PYTHONPATH:+:${PYTHONPATH}}"
export GRPO_SITE_DIR="${site_dir}"

log "installing staged GRPO runtime wheels into ${site_dir} (offline)"
"${pip_cmd[@]}" install --target "${site_dir}" --no-index --find-links "${wheelhouse}" --no-deps --upgrade \
    "${wheelhouse}/accelerate-1.3.0-py3-none-any.whl" \
    "${wheelhouse}/datasets-3.2.0-py3-none-any.whl" \
    "${wheelhouse}/trl-0.16.0-py3-none-any.whl" \
    "${wheelhouse}/peft-0.14.0-py3-none-any.whl" \
    "${wheelhouse}/nltk-3.9.1-py3-none-any.whl" \
    "${wheelhouse}/einops-0.8.0-py3-none-any.whl" \
    "${wheelhouse}/pyarrow-18.1.0-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "${wheelhouse}/pandas-2.2.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/dill-0.3.8-py3-none-any.whl" \
    "${wheelhouse}/multiprocess-0.70.16-py312-none-any.whl" \
    "${wheelhouse}/xxhash-3.5.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/hjson-3.1.0-py3-none-any.whl" \
    "${wheelhouse}/ninja-1.11.1.3-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl" \
    "${wheelhouse}/nvidia_ml_py-12.560.30-py3-none-any.whl" \
    "${wheelhouse}/py_cpuinfo-9.0.0-py3-none-any.whl" \
    "${wheelhouse}/pydantic-2.10.5-py3-none-any.whl" \
    "${wheelhouse}/pydantic_core-2.27.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/annotated_types-0.7.0-py3-none-any.whl" \
    "${wheelhouse}/typing_extensions-4.12.2-py3-none-any.whl" \
    "${wheelhouse}/rich-13.9.4-py3-none-any.whl" \
    "${wheelhouse}/markdown_it_py-3.0.0-py3-none-any.whl" \
    "${wheelhouse}/mdurl-0.1.2-py3-none-any.whl" \
    "${wheelhouse}/absl_py-2.1.0-py3-none-any.whl" \
    "${wheelhouse}/sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/fsspec-2024.9.0-py3-none-any.whl" \
    "${wheelhouse}/aiohttp-3.11.11-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/aiohappyeyeballs-2.4.4-py3-none-any.whl" \
    "${wheelhouse}/aiosignal-1.3.2-py2.py3-none-any.whl" \
    "${wheelhouse}/async_timeout-5.0.1-py3-none-any.whl" \
    "${wheelhouse}/attrs-24.3.0-py3-none-any.whl" \
    "${wheelhouse}/frozenlist-1.5.0-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/multidict-6.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/propcache-0.2.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/yarl-1.18.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/python_dateutil-2.9.0.post0-py2.py3-none-any.whl" \
    "${wheelhouse}/pytz-2024.2-py2.py3-none-any.whl" \
    "${wheelhouse}/tzdata-2024.2-py2.py3-none-any.whl" \
    "${wheelhouse}/six-1.17.0-py2.py3-none-any.whl" \
    "${wheelhouse}/docstring_parser-0.16-py3-none-any.whl" \
    "${wheelhouse}/shtab-1.7.1-py3-none-any.whl" \
    "${wheelhouse}/colorama-0.4.6-py2.py3-none-any.whl" \
    "${wheelhouse}/regex-2024.11.6-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${wheelhouse}/joblib-1.4.2-py3-none-any.whl"

log "installing source distributions without CUDA extension builds"
DS_BUILD_OPS=0 "${pip_cmd[@]}" install --target "${site_dir}" --no-index --no-deps --no-build-isolation --upgrade \
    "${wheelhouse}/deepspeed-0.15.4.tar.gz" \
    "${wheelhouse}/rouge_score-0.1.2.tar.gz"

log "pinning the Video-R1 checkpoint-compatible transformers stack"
"${pip_cmd[@]}" install --target "${site_dir}" --no-index --no-deps --no-build-isolation --upgrade --force-reinstall \
    "${compat_wheelhouse}/tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${compat_wheelhouse}/av-14.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${compat_wheelhouse}/transformers-4.49.0.dev0-py3-none-any.whl"

log "running import/version gate"
"${python_bin}" - <<'PY'
import importlib.metadata as metadata
import os
import torch
import torchvision
import flash_attn
import transformers
import tokenizers
import accelerate
import datasets
import deepspeed
import trl
import peft
import nltk
import rouge_score
import qwen_vl_utils
from transformers import Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.grpo import GRPOScriptArguments
from trainer.video_ktr_grpo_trainer import VideoKTRGRPOTrainer

expected = {
    "transformers": "4.49.0.dev0",
    "tokenizers": "0.21.4",
    "trl": "0.16.0",
    "deepspeed": "0.15.4",
    "datasets": "3.2.0",
    "accelerate": "1.3.0",
    "peft": "0.14.0",
}
observed = {
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "trl": trl.__version__,
    "deepspeed": deepspeed.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "peft": peft.__version__,
}
for name, version in expected.items():
    if observed[name] != version:
        raise SystemExit(f"version mismatch for {name}: {observed[name]} != {version}")
print("[grpo-env] versions=" + repr(observed))
print("[grpo-env] torch=" + torch.__version__ + " cuda=" + str(torch.version.cuda))
print("[grpo-env] torchvision=" + torchvision.__version__ + " flash_attn=" + flash_attn.__version__)
print("[grpo-env] transformers_file=" + transformers.__file__)
print("[grpo-env] trainer=" + VideoKTRGRPOTrainer.__module__)
site_dir = os.environ["GRPO_SITE_DIR"]
if not transformers.__file__.startswith(site_dir):
    raise SystemExit("transformers did not resolve from the GRPO site overlay")
if not tokenizers.__file__.startswith(site_dir):
    raise SystemExit("tokenizers did not resolve from the GRPO site overlay")
if torch.__file__.startswith(site_dir):
    raise SystemExit("torch must remain supplied by the GPU image, not the overlay")
if metadata.version("msgpack") is None:
    raise SystemExit("msgpack is unavailable")
PY

log "PASS: isolated GRPO site overlay is ready: ${site_dir}"
