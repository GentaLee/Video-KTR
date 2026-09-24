#!/usr/bin/env bash
# Backward-compatible spelling for the requested smoke launcher.
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${project_root}/smoke.sh" "$@"
