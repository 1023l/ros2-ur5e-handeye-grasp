#!/usr/bin/env python3
"""Pick & place CLI (uses shared pick_place_core)."""

from __future__ import annotations

import sys

import rclpy

from arm_system.moveit_client import MoveItArmClient
from arm_system.pick_place_core import GripperClient, TargetWaiter, run_pick_place, wait_target_pose


def main() -> int:
    rclpy.init()
    waiter = TargetWaiter()
    gripper = GripperClient()
    arm = MoveItArmClient(node_name="arm_pick_place_moveit")
    try:
        waiter.get_logger().info("Waiting for /arm/target_pose_base ...")
        pose = wait_target_pose(waiter, 20.0)
        if pose is None:
            waiter.get_logger().error(
                "No target. Run: ros2 launch arm_system arm_vision_demo.launch.py"
            )
            return 2
        if not arm.wait_ready(10.0):
            return 2
        return run_pick_place(arm, gripper, pose)
    finally:
        waiter.destroy_node()
        gripper.destroy_node()
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
