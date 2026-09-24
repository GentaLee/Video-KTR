#!/usr/bin/env bash
# Fresh baseline on the validated B200 release; never resume a KTR checkpoint.
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_root="$(dirname "${project_root}")"
if [[ "$(basename "$(dirname "${release_root}")")" == releases ]]; then
    default_root="$(dirname "$(dirname "${release_root}")")"
else
    default_root="${release_root}"
fi
export B200_ROOT="${B200_ROOT:-${default_root}}"
export B200_ENV_FILE="${B200_ENV_FILE:-${B200_ROOT}/b200.env}"
if [[ -f "${release_root}/environment-manifest.json" ]]; then
    default_manifest="${release_root}/environment-manifest.json"
else
    default_manifest="${B200_ROOT}/environment-manifest.json"
fi
export B200_ENV_MANIFEST="${B200_ENV_MANIFEST:-${default_manifest}}"
export B200_SMOKE_RUN_ROOT="${B200_SMOKE_RUN_ROOT:-${B200_ROOT}/artifacts/grpo-smoke-b200/tail-fix-86317d0}"
# run_full preserves this explicit empty value over any private env setting.
export B200_RESUME_FROM_CHECKPOINT=
printf 'Fresh B200 baseline; checkpoint resume disabled. Source: %s\n' "${project_root}"
exec bash "${project_root}/scripts/launch_b200.sh" baseline
