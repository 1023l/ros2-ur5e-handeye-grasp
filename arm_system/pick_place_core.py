"""Shared pick-and-place routine used by CLI and agent skills."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_srvs.srv import SetBool

from arm_system.moveit_client import SUCCESS, MoveItArmClient
from arm_system.pose_utils import tool_down_pose


class TargetWaiter(Node):
    def __init__(self, node_name: str = "arm_target_waiter") -> None:
        super().__init__(node_name)
        self.pose: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, "/arm/target_pose_base", self._cb, 10)

    def _cb(self, msg: PoseStamped) -> None:
        if self.pose is None:
            self.pose = msg
            p = msg.pose.position
            self.get_logger().info(f"Pick target xyz=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")


class GripperClient(Node):
    def __init__(self, node_name: str = "arm_gripper_client") -> None:
        super().__init__(node_name)
        self._cli = self.create_client(SetBool, "/arm/gripper/set")

    def set_closed(self, closed: bool, timeout: float = 5.0) -> bool:
        if not self._cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("/arm/gripper/set not available")
            return False
        req = SetBool.Request()
        req.data = closed
        fut = self._cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        res = fut.result()
        if res is None or not res.success:
            self.get_logger().error("Gripper service failed")
            return False
        time.sleep(0.3)
        return True


def wait_target_pose(
    waiter: TargetWaiter, timeout_sec: float = 20.0
) -> Optional[PoseStamped]:
    deadline = time.time() + timeout_sec
    while waiter.pose is None and rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(waiter, timeout_sec=0.1)
    return waiter.pose


def _ok(arm: MoveItArmClient, code: int, step: str) -> bool:
    if code == SUCCESS:
        arm.get_logger().info(f"[OK] {step}")
        return True
    arm.get_logger().error(f"[FAIL] {step} error_code={code}")
    return False


def run_pick_place(
    arm: MoveItArmClient,
    gripper: GripperClient,
    target: PoseStamped,
    *,
    approach_z: float = 0.22,
    grasp_z: float = 0.10,
    place_xy: Tuple[float, float] = (0.30, 0.28),
    place_z: float = 0.10,
    vel: float = 0.22,
    acc: float = 0.22,
    max_step_retries: int = 1,
) -> int:
    """Execute full pick→place→home. Returns 0 on success.

    Each motion step may retry once (recovery depth track C).
    """
    px = float(target.pose.position.x)
    py = float(target.pose.position.y)
    place_x, place_y = place_xy

    arm.get_logger().info("=== PICK & PLACE start ===")

    if not gripper.set_closed(False):
        return 2

    def move(name: str, pose, v: float, a: float) -> bool:
        for attempt in range(max_step_retries + 1):
            code = arm.move_pose(
                pose, velocity_scale=v, accel_scale=a, planning_time=10.0
            )
            if _ok(arm, code, f"{name}" + (f" retry{attempt}" if attempt else "")):
                return True
            arm.get_logger().warn(f"step '{name}' failed attempt={attempt}")
        return False

    steps = [
        ("approach pick", tool_down_pose(px, py, approach_z), vel, acc),
        ("descend pick", tool_down_pose(px, py, grasp_z), 0.15, 0.15),
    ]
    for name, pose, v, a in steps:
        if not move(name, pose, v, a):
            return 1

    if not gripper.set_closed(True):
        return 2
    arm.get_logger().info("Grasped (fake)")

    more = [
        ("lift", tool_down_pose(px, py, approach_z), vel, acc),
        ("approach place", tool_down_pose(place_x, place_y, approach_z), vel, acc),
        ("descend place", tool_down_pose(place_x, place_y, place_z), 0.15, 0.15),
    ]
    for name, pose, v, a in more:
        if not move(name, pose, v, a):
            return 1

    if not gripper.set_closed(False):
        return 2
    arm.get_logger().info("Released (fake)")

    if not move("retreat", tool_down_pose(place_x, place_y, approach_z), vel, acc):
        return 1

    for attempt in range(max_step_retries + 1):
        code = arm.move_named("home", velocity_scale=0.3, accel_scale=0.3)
        if _ok(arm, code, "home" + (f" retry{attempt}" if attempt else "")):
            arm.get_logger().info("=== PICK & PLACE SUCCESS ===")
            return 0
    return 1


def run_approach(
    arm: MoveItArmClient,
    target: PoseStamped,
    *,
    approach_z: float = 0.22,
) -> int:
    pose = tool_down_pose(
        float(target.pose.position.x),
        float(target.pose.position.y),
        approach_z,
    )
    arm.get_logger().info(
        f"Approaching xyz=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
    )
    return 0 if _ok(
        arm,
        arm.move_pose(pose, velocity_scale=0.25, accel_scale=0.25, planning_time=10.0),
        "approach",
    ) else 1
