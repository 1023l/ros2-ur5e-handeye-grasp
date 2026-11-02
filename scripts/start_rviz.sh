#!/bin/bash
source /opt/ros/humble/setup.bash
if [[ -f "${HOME}/ros2_ws/install/setup.bash" ]]; then
  source "${HOME}/ros2_ws/install/setup.bash"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CFG=$(ros2 pkg prefix ur_moveit_config)/share/ur_moveit_config/rviz/view_robot.rviz
echo "Starting rviz with $RVIZ_CFG"
exec "${SCRIPT_DIR}/rviz2_wslg.sh" -d "$RVIZ_CFG"
