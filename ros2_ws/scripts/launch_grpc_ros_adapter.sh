#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${WORKSPACE_DIR}/.venv"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Virtual environment not found at ${VENV_DIR}." >&2
  echo "Run ${WORKSPACE_DIR}/scripts/bootstrap_uv_env.sh first." >&2
  exit 1
fi

if [ ! -f "${WORKSPACE_DIR}/install/setup.bash" ]; then
  echo "Workspace install/setup.bash not found." >&2
  echo "Run ${WORKSPACE_DIR}/scripts/colcon_build_uv.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_ros2_uv.sh" --

set +e
ros2 launch grpc_ros_adapter ros2_server_launch.py "$@"
STATUS=$?
set -e

if [ "${STATUS}" -ne 0 ]; then
  echo
  echo "grpc_ros_adapter launch exited with status ${STATUS}." >&2
  if [ -t 0 ]; then
    read -r -p "Press Enter to close this terminal..." _
  fi
fi

exit "${STATUS}"
