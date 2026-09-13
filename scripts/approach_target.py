#!/usr/bin/env python3
"""Approach above /arm/target_pose_base (shared core)."""

from __future__ import annotations

import sys

import rclpy

from arm_system.moveit_client import MoveItArmClient
from arm_system.pick_place_core import TargetWaiter, run_approach, wait_target_pose


def main() -> int:
    rclpy.init()
    waiter = TargetWaiter(node_name="arm_approach_target")
    arm = MoveItArmClient(node_name="arm_approach_moveit")
    try:
        waiter.get_logger().info("Waiting for /arm/target_pose_base ...")
        pose = wait_target_pose(waiter, 15.0)
        if pose is None:
            waiter.get_logger().error(
                "Timeout: ros2 launch arm_system arm_vision_demo.launch.py"
            )
            return 2
        if not arm.wait_ready(10.0):
            return 2
        return run_approach(arm, pose)
    finally:
        waiter.destroy_node()
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
