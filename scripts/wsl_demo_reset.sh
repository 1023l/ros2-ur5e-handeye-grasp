#!/usr/bin/env bash
# Stop stale ROS nodes (common when MoveIt was launched twice → 2× /move_action).
set -euo pipefail
pkill -2 -f move_group 2>/dev/null || true
pkill -2 -f ros2_control_node 2>/dev/null || true
pkill -2 -f rviz2 2>/dev/null || true
pkill -2 -f "arm_system/(arm_|policy_dock|pick_place)" 2>/dev/null || true
sleep 2
pkill -9 -f move_group 2>/dev/null || true
pkill -9 -f ros2_control_node 2>/dev/null || true
ros2 daemon stop 2>/dev/null || true
echo "Demo reset done. Start terminals fresh (see docs/WSL_ROS2.md)."
