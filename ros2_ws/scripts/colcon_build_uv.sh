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

source /opt/ros/humble/setup.bash
source "${VENV_DIR}/bin/activate"

export FINSSIM_REPO_ROOT="$(cd -- "${WORKSPACE_DIR}/.." && pwd)"
export COLCON_PYTHON_EXECUTABLE="${VENV_DIR}/bin/python"
export PYTHON_EXECUTABLE="${VENV_DIR}/bin/python"
export PYTHONNOUSERSITE=1

# This machine has Homebrew binutils in the interactive shell. If
# /home/linuxbrew/.linuxbrew/bin/ld is used for ROS2 packages, it fails to
# resolve Ubuntu/ROS transitive libraries required by /usr/local OpenCV 4.12.
# Keep /usr/local OpenCV, but force the native Ubuntu toolchain/linker.
export PATH="${VENV_DIR}/bin:/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export LD=/usr/bin/ld
export OpenCV_DIR="/usr/local/lib/cmake/opencv4"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig"
export LD_LIBRARY_PATH="/usr/local/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

VENV_SITE_PACKAGES="$("${VENV_DIR}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="${VENV_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${WORKSPACE_DIR}"
colcon build \
  "$@" \
  --cmake-args \
  -DPython3_EXECUTABLE="${VENV_DIR}/bin/python" \
  -DPYTHON_EXECUTABLE="${VENV_DIR}/bin/python" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_LINKER=/usr/bin/ld \
  -DOpenCV_DIR="${OpenCV_DIR}"

while IFS= read -r -d '' script_path; do
  if ! grep -Iq . "${script_path}"; then
    continue
  fi
  first_line="$(head -n 1 "${script_path}" || true)"
  if [[ "${first_line}" == "#!/usr/bin/python3" ]]; then
    sed -i "1s|^#!/usr/bin/python3$|#!${VENV_DIR}/bin/python|" "${script_path}"
  fi
done < <(find "${WORKSPACE_DIR}/install" -type f -path "*/lib/*/*" -perm -111 -print0)
