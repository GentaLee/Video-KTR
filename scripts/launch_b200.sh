#!/usr/bin/env bash
# Detached strict training with a disposable terminal log viewer.
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant="${1:-}"
if [[ "$#" != 1 || ( "${variant}" != ktr && "${variant}" != baseline ) ]]; then
    printf 'Usage: bash launch_b200.sh ktr|baseline\n' >&2
    exit 2
fi
b200_root="${B200_ROOT:-$(dirname "${project_root}")}"
env_file="${B200_ENV_FILE:-${b200_root}/b200.env}"
if [[ -z "${B200_ROOT:-}" && -f "${env_file}" ]]; then
    # Read only the root in a subshell. Preserve every exported override for
    # run_full's own merge with b200.env (especially timeout/cache/smoke paths).
    b200_root="$(
        source "${env_file}"
        printf '%s' "${B200_ROOT:-${b200_root}}"
    )"
fi
stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
launch_dir="${b200_root}/artifacts/launches"
mkdir -p "${launch_dir}"
launch_log="${launch_dir}/${variant}-${stamp}.log"
exit_file="${launch_dir}/${variant}-${stamp}.exit"
run_root="${b200_root}/artifacts/grpo-full-b200/${variant}-${stamp}"
touch "${launch_log}"

nohup setsid env B200_ROOT="${b200_root}" B200_ENV_FILE="${env_file}" \
    B200_RUN_ROOT="${run_root}" VARIANT="${variant}" MAX_STEPS=-1 \
    B200_ALLOW_PAUSE_KEEPALIVE=1 B200_ALLOW_REDUCED_HOLMES=0 B200_ALLOW_DECODER_FILTERED=0 \
    bash -c 'set +e; bash "$1"; result=$?; printf "%s\n" "$result" > "$2"; exit "$result"' \
    bash "${project_root}/scripts/run_full_b200.sh" "${exit_file}" \
    > "${launch_log}" 2>&1 < /dev/null &
job_pid="$!"
printf 'Submitted %s (launcher PID %s). Keep-alive is managed automatically.\nRun: %s\nLog: %s\nCtrl-C closes this viewer; the training continues.\n' \
    "${variant}" "${job_pid}" "${run_root}" "${launch_log}"
trap 'exit 0' INT TERM HUP
tail --pid="${job_pid}" --sleep-interval=1 -n +1 -F "${launch_log}"
if [[ -f "${exit_file}" ]]; then
    result="$(<"${exit_file}")"
    printf 'Launcher exited with status %s. Log: %s\n' "${result}" "${launch_log}"
    exit "${result}"
fi
printf 'Launcher exited without an exit record; inspect %s\n' "${launch_log}" >&2
exit 1
