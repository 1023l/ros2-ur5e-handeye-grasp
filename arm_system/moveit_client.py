#!/usr/bin/env python3
"""UR5e MoveIt2 thin client (Humble): named / joint / pose goals."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import rclpy
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


# Matches ur_moveit_config SRDF group_state values (prefix empty, name=ur).
NAMED_JOINT_TARGETS: Dict[str, Dict[str, float]] = {
    "home": {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.5707,
        "elbow_joint": 0.0,
        "wrist_1_joint": 0.0,
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
    },
    "up": {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.5707,
        "elbow_joint": 0.0,
        "wrist_1_joint": -1.5707,
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
    },
}

ARM_JOINTS: Tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# MoveIt MoveItErrorCodes
SUCCESS = 1


@dataclass
class PlanMetrics:
    """Plan-only outcome for Pareto / tradeoff curves."""

    success: bool
    error_code: int
    wall_time_s: float
    traj_duration_s: float
    path_length_rad: float
    n_waypoints: int


def _traj_metrics(result) -> Tuple[float, float, int]:
    """Extract (duration_s, path_length_rad, n_waypoints) from MoveGroup result."""
    try:
        jt = result.planned_trajectory.joint_trajectory
    except Exception:
        return 0.0, 0.0, 0
    pts = jt.points
    if not pts:
        return 0.0, 0.0, 0
    last = pts[-1].time_from_start
    duration = float(last.sec) + float(last.nanosec) * 1e-9
    path = 0.0
    prev = None
    for p in pts:
        q = list(p.positions)
        if prev is not None and len(q) == len(prev):
            path += float(math.sqrt(sum((a - b) ** 2 for a, b in zip(q, prev))))
        prev = q
    return duration, path, len(pts)


class MoveItArmClient(Node):
    """Plan (+ optionally execute) via /move_action for ur_manipulator."""

    def __init__(
        self,
        group_name: str = "ur_manipulator",
        ee_link: str = "tool0",
        base_frame: str = "base_link",
        node_name: str = "arm_moveit_client",
    ) -> None:
        super().__init__(node_name)
        self.group_name = group_name
        self.ee_link = ee_link
        self.base_frame = base_frame
        self._ac = ActionClient(self, MoveGroup, "move_action")

    def wait_ready(self, timeout_sec: float = 20.0) -> bool:
        self.get_logger().info("Waiting for /move_action ...")
        ok = self._ac.wait_for_server(timeout_sec=timeout_sec)
        if not ok:
            self.get_logger().error("move_action server not available")
        return ok

    def move_named(
        self,
        name: str,
        *,
        execute: bool = True,
        velocity_scale: float = 0.3,
        accel_scale: float = 0.3,
    ) -> int:
        key = name.strip().lower()
        if key not in NAMED_JOINT_TARGETS:
            raise KeyError(
                f"Unknown named pose '{name}'. Known: {sorted(NAMED_JOINT_TARGETS)}"
            )
        return self.move_joints(
            NAMED_JOINT_TARGETS[key],
            execute=execute,
            velocity_scale=velocity_scale,
            accel_scale=accel_scale,
        )

    def move_joints(
        self,
        joints: Mapping[str, float],
        *,
        execute: bool = True,
        velocity_scale: float = 0.3,
        accel_scale: float = 0.3,
        planning_time: float = 5.0,
        tolerance: float = 0.01,
    ) -> int:
        cons = Constraints()
        for jn in ARM_JOINTS:
            if jn not in joints:
                raise KeyError(f"Missing joint '{jn}' in target")
            jc = JointConstraint()
            jc.joint_name = jn
            jc.position = float(joints[jn])
            jc.tolerance_above = tolerance
            jc.tolerance_below = tolerance
            jc.weight = 1.0
            cons.joint_constraints.append(jc)
        return self._send(cons, execute, velocity_scale, accel_scale, planning_time)

    def move_pose(
        self,
        pose: Pose,
        *,
        frame_id: Optional[str] = None,
        execute: bool = True,
        velocity_scale: float = 0.2,
        accel_scale: float = 0.2,
        planning_time: float = 8.0,
        position_tol: float = 0.01,
        orientation_tol: float = 0.1,
    ) -> int:
        """Cartesian goal for ee_link in base_frame (or frame_id)."""
        m = self.plan_pose_metrics(
            pose,
            frame_id=frame_id,
            execute=execute,
            velocity_scale=velocity_scale,
            accel_scale=accel_scale,
            planning_time=planning_time,
            position_tol=position_tol,
            orientation_tol=orientation_tol,
        )
        return m.error_code

    def plan_pose_metrics(
        self,
        pose: Pose,
        *,
        frame_id: Optional[str] = None,
        execute: bool = False,
        velocity_scale: float = 0.2,
        accel_scale: float = 0.2,
        planning_time: float = 8.0,
        position_tol: float = 0.01,
        orientation_tol: float = 0.1,
        num_attempts: int = 10,
    ) -> PlanMetrics:
        """Plan (default plan-only) and return duration / path-length metrics."""
        frame = frame_id or self.base_frame
        cons = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = frame
        pc.link_name = self.ee_link
        pc.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [position_tol]
        pc.constraint_region.primitives.append(sphere)
        pc.constraint_region.primitive_poses.append(pose)
        cons.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = frame
        oc.link_name = self.ee_link
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = orientation_tol
        oc.absolute_y_axis_tolerance = orientation_tol
        oc.absolute_z_axis_tolerance = orientation_tol
        oc.weight = 1.0
        cons.orientation_constraints.append(oc)

        return self._send_metrics(
            cons,
            execute,
            velocity_scale,
            accel_scale,
            planning_time,
            num_attempts=num_attempts,
        )

    def move_pose_rpy(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        **kwargs,
    ) -> int:
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation = _quat_from_rpy(roll, pitch, yaw)
        return self.move_pose(pose, **kwargs)

    def _send(
        self,
        goal_constraints: Constraints,
        execute: bool,
        velocity_scale: float,
        accel_scale: float,
        planning_time: float,
    ) -> int:
        return self._send_metrics(
            goal_constraints, execute, velocity_scale, accel_scale, planning_time
        ).error_code

    def _send_metrics(
        self,
        goal_constraints: Constraints,
        execute: bool,
        velocity_scale: float,
        accel_scale: float,
        planning_time: float,
        *,
        num_attempts: int = 10,
    ) -> PlanMetrics:
        if not self._ac.server_is_ready():
            if not self.wait_ready():
                return PlanMetrics(False, -1, 0.0, 0.0, 0.0, 0)

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.num_planning_attempts = int(num_attempts)
        req.allowed_planning_time = planning_time
        req.max_velocity_scaling_factor = velocity_scale
        req.max_acceleration_scaling_factor = accel_scale
        req.goal_constraints.append(goal_constraints)
        goal.request = req

        opts = PlanningOptions()
        opts.plan_only = not execute
        goal.planning_options = opts

        self.get_logger().info(
            f"MoveGroup plan{'+execute' if execute else ' only'} "
            f"(group={self.group_name})"
        )
        t0 = time.perf_counter()
        send_future = self._ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Goal rejected by move_group")
            return PlanMetrics(False, -1, time.perf_counter() - t0, 0.0, 0.0, 0)

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wall = time.perf_counter() - t0
        result = result_future.result().result
        code = int(result.error_code.val)
        dur, path, n_wp = _traj_metrics(result)
        ok = code == SUCCESS
        if ok:
            self.get_logger().info(
                f"SUCCESS wall={wall:.2f}s traj={dur:.2f}s path={path:.3f}rad"
            )
        else:
            self.get_logger().error(f"Failed, MoveIt error_code={code}")
        return PlanMetrics(ok, code, wall, dur, path, n_wp)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def list_named() -> List[str]:
    return sorted(NAMED_JOINT_TARGETS.keys())
