"""Shared pose helpers for arm demos."""

from __future__ import annotations

import math

from geometry_msgs.msg import Pose, Quaternion


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def tool_down_pose(x: float, y: float, z: float) -> Pose:
    """tool0 roughly pointing down (table pick orientation)."""
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation = quat_from_rpy(math.pi, 0.0, 0.0)
    return pose
