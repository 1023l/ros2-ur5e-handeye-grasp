"""Skill executor: maps skill ids to MoveIt / gripper / pick-place."""

from __future__ import annotations

from typing import Optional

import rclpy

from arm_system.intent import list_skills_text
from arm_system.moveit_client import SUCCESS, MoveItArmClient
from arm_system.pick_place_core import (
    GripperClient,
    TargetWaiter,
    run_approach,
    run_pick_place,
    wait_target_pose,
)


class SkillExecutor:
    def __init__(self, logger=None) -> None:
        self._log = logger
        self.arm = MoveItArmClient(node_name="arm_agent_moveit")
        self.gripper = GripperClient(node_name="arm_agent_gripper")
        self._ready = False

    def _info(self, msg: str) -> None:
        if self._log:
            self._log.info(msg)
        else:
            self.arm.get_logger().info(msg)

    def ensure_moveit(self, timeout: float = 10.0) -> bool:
        if self._ready:
            return True
        self._ready = self.arm.wait_ready(timeout)
        return self._ready

    def destroy(self) -> None:
        self.arm.destroy_node()
        self.gripper.destroy_node()

    def run(self, skill_id: str) -> tuple[int, str]:
        if skill_id == "help":
            return 0, list_skills_text()

        if skill_id in ("home", "up", "approach", "pick_place"):
            if not self.ensure_moveit():
                return 2, "move_action not available (start ur5e_moveit_demo)"

        if skill_id == "home":
            code = self.arm.move_named("home", velocity_scale=0.3, accel_scale=0.3)
            return (0, "home OK") if code == SUCCESS else (1, f"home failed code={code}")

        if skill_id == "up":
            code = self.arm.move_named("up", velocity_scale=0.3, accel_scale=0.3)
            return (0, "up OK") if code == SUCCESS else (1, f"up failed code={code}")

        if skill_id == "gripper_open":
            ok = self.gripper.set_closed(False)
            return (0, "gripper open") if ok else (2, "gripper service failed")

        if skill_id == "gripper_close":
            ok = self.gripper.set_closed(True)
            return (0, "gripper close") if ok else (2, "gripper service failed")

        if skill_id == "approach":
            waiter = TargetWaiter(node_name="arm_agent_approach_waiter")
            try:
                self._info("Waiting /arm/target_pose_base ...")
                pose = wait_target_pose(waiter, 20.0)
                if pose is None:
                    return 2, "no target_pose_base (start arm_vision_demo)"
                code = run_approach(self.arm, pose)
                return (0, "approach OK") if code == 0 else (code, "approach failed")
            finally:
                waiter.destroy_node()

        if skill_id == "pick_place":
            waiter = TargetWaiter(node_name="arm_agent_pp_waiter")
            try:
                self._info("Waiting /arm/target_pose_base for pick_place ...")
                pose = wait_target_pose(waiter, 20.0)
                if pose is None:
                    return 2, "no target_pose_base (start arm_vision_demo)"
                code = run_pick_place(self.arm, self.gripper, pose)
                return (0, "pick_place OK") if code == 0 else (code, "pick_place failed")
            finally:
                waiter.destroy_node()

        return 2, f"unknown skill: {skill_id}"
