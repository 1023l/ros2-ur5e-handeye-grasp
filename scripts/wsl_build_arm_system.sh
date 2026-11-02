#!/usr/bin/env bash
# Run inside: wsl -d Ubuntu-22.04
set -euo pipefail
PKG="/mnt/c/Users/Administrator/Documents/trae_projects/arm_system"
source /opt/ros/humble/setup.bash
echo "ROS: ${ROS_DISTRO:-unknown}"
python3 -c "import torch; print('torch', torch.__version__)" || echo "WARN: install torch in WSL if training here"
export PYTHONPATH="${PKG}:${PYTHONPATH:-}"
python3 -c "import arm_system.policy.act_model; print('arm_system.policy OK')"
if [ -d "${HOME}/ros2_ws/src" ]; then
  ln -sfn "${PKG}" "${HOME}/ros2_ws/src/arm_system" 2>/dev/null || true
  cd "${HOME}/ros2_ws"
  colcon build --packages-select arm_system --symlink-install
  echo "Built in ~/ros2_ws — source install/setup.bash"
else
  echo "No ~/ros2_ws — using PYTHONPATH-only mode (scripts still work)"
fi
