#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: ./scripts/run_ros2_uv.sh [--skip-install-setup] <command> [args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ARGS=()

if [ "${1}" = "--skip-install-setup" ]; then
  SOURCE_ARGS+=(--skip-install-setup)
  shift
fi

if [ "$#" -eq 0 ]; then
  echo "No command provided." >&2
  exit 1
fi

CMD=("$@")

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_ros2_uv.sh" "${SOURCE_ARGS[@]}" --

set +e
"${CMD[@]}"
STATUS=$?
set -e

if [ "${STATUS}" -ne 0 ]; then
  echo
  echo "Command exited with status ${STATUS}: ${CMD[*]}" >&2
  if [ -t 0 ]; then
    read -r -p "Press Enter to close this terminal..." _
  fi
fi

exit "${STATUS}"
