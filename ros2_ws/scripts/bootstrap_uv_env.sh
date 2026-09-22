#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${WORKSPACE_DIR}/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

if [ -f "${VENV_DIR}/pyvenv.cfg" ] && grep -q '^include-system-site-packages = true$' "${VENV_DIR}/pyvenv.cfg"; then
  echo "Recreating ${VENV_DIR} without system-site-packages to avoid ROS/user-site Python conflicts."
  rm -rf "${VENV_DIR}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  uv venv "${VENV_DIR}" --python 3.10
fi

uv sync --project "${WORKSPACE_DIR}"

cat <<EOF
Workspace uv environment is ready at:
  ${VENV_DIR}

Next steps:
  cd ${WORKSPACE_DIR}
  ./scripts/colcon_build_uv.sh
  source ./scripts/source_ros2_uv.sh
  ./scripts/run_ros2_uv.sh ros2 launch grpc_ros_adapter ros2_server_launch.py
EOF
