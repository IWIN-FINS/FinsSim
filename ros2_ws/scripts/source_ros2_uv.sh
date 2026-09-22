#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:" >&2
  echo "  source ./scripts/source_ros2_uv.sh" >&2
  exit 1
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${WORKSPACE_DIR}/.." && pwd)"
VENV_DIR="${WORKSPACE_DIR}/.venv"
SKIP_INSTALL_SETUP=0

for arg in "$@"; do
  case "${arg}" in
    --)
      ;;
    --skip-install-setup)
      SKIP_INSTALL_SETUP=1
      ;;
    *)
      echo "Unknown option for source_ros2_uv.sh: ${arg}" >&2
      return 1
      ;;
  esac
done

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Virtual environment not found at ${VENV_DIR}." >&2
  echo "Run ${WORKSPACE_DIR}/scripts/bootstrap_uv_env.sh first." >&2
  return 1
fi

set +u
source /opt/ros/humble/setup.bash
source "${VENV_DIR}/bin/activate"
set -u

export FINSSIM_REPO_ROOT="${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export COLCON_PYTHON_EXECUTABLE="${VENV_DIR}/bin/python"
export PYTHON_EXECUTABLE="${VENV_DIR}/bin/python"

# Use the same native ROS/OpenCV runtime as the build wrapper. The interactive
# shell may put Homebrew binutils and old OpenCV paths before the Ubuntu/ROS
# stack; do not let those leak into ROS2 nodes.
export PATH="${VENV_DIR}/bin:/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export OpenCV_DIR="/usr/local/lib/cmake/opencv4"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig"
export LD_LIBRARY_PATH="/usr/local/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

# Some desktop sessions persist CUDA_VISIBLE_DEVICES as a GPU UUID from a
# previous machine/session. If that UUID is stale, PyTorch reports zero CUDA
# devices even though nvidia-smi can see GPUs. Clear only invalid UUID masks;
# numeric masks such as "0" or "1" are left untouched.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  if [[ "${CUDA_VISIBLE_DEVICES}" == GPU-* ]]; then
    if ! nvidia-smi -L 2>/dev/null | grep -Fq "${CUDA_VISIBLE_DEVICES}"; then
      echo "Warning: clearing stale CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
      unset CUDA_VISIBLE_DEVICES
    fi
  fi
fi

VENV_SITE_PACKAGES="$("${VENV_DIR}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
case ":${PYTHONPATH:-}:" in
  *":${VENV_SITE_PACKAGES}:"*)
    ;;
  *)
    export PYTHONPATH="${VENV_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
    ;;
esac

if [ "${SKIP_INSTALL_SETUP}" -eq 0 ]; then
  if [ ! -f "${WORKSPACE_DIR}/install/setup.bash" ]; then
    echo "Workspace install/setup.bash not found." >&2
    echo "Run ${WORKSPACE_DIR}/scripts/colcon_build_uv.sh first." >&2
    return 1
  fi
  set +u
  source "${WORKSPACE_DIR}/install/setup.bash"
  set -u
fi
