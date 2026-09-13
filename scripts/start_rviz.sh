#!/bin/bash
source /opt/ros/humble/setup.bash
source /home/inspect/inspect_ws/install/setup.bash
export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
RVIZ_CFG=$(ros2 pkg prefix ur_moveit_config)/share/ur_moveit_config/rviz/view_robot.rviz
echo "Starting rviz with $RVIZ_CFG"
exec rviz2 -d "$RVIZ_CFG"
