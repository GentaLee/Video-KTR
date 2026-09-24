#!/usr/bin/env bash
# Single operator entry. No job is started without an explicit subcommand.
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$#" != 1 ]]; then
    printf 'Usage: bash run.sh baseline|ktr|smoke|--help\n' >&2
    exit 2
fi
case "$1" in
    --help|-h) printf 'Usage: bash run.sh baseline|ktr|smoke\nAll jobs manage keep-alive; baseline always starts fresh.\n'; exit 0 ;;
    baseline|ktr|smoke) ;;
    *) printf 'Unknown command: %s\n' "$1" >&2; exit 2 ;;
esac
release_root="$(dirname "${project_root}")"
if [[ "$(basename "$(dirname "${release_root}")")" == releases ]]; then
    default_root="$(dirname "$(dirname "${release_root}")")"
else
    default_root="${release_root}"
fi
export B200_ROOT="${B200_ROOT:-${default_root}}"
export B200_ENV_FILE="${B200_ENV_FILE:-${B200_ROOT}/b200.env}"
if [[ -f "${release_root}/environment-manifest.json" ]]; then
    export B200_ENV_MANIFEST="${B200_ENV_MANIFEST:-${release_root}/environment-manifest.json}"
fi
export B200_SMOKE_RUN_ROOT="${B200_SMOKE_RUN_ROOT:-${B200_ROOT}/artifacts/grpo-smoke-b200/tail-fix-86317d0}"
case "$1" in
    baseline) exec bash "${project_root}/scripts/run_baseline_b200.sh" ;;
    ktr) exec bash "${project_root}/scripts/launch_b200.sh" ktr ;;
    smoke)
        export B200_ALLOW_PAUSE_KEEPALIVE=1
        exec bash "${project_root}/scripts/somke-b200.sh"
        ;;
esac
