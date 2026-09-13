"""Apply simple collision obstacles into MoveIt planning scene (depth track B)."""

from __future__ import annotations

from typing import Optional

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


def add_box_obstacle(
    node: Node,
    *,
    object_id: str = "table_clutter_box",
    frame_id: str = "base_link",
    xyz=(0.35, 0.12, 0.08),
    size=(0.08, 0.08, 0.16),
    timeout: float = 5.0,
) -> bool:
    """Insert an axis-aligned box obstacle; returns False if service missing."""
    cli = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    if not cli.wait_for_service(timeout_sec=timeout):
        node.get_logger().warn("/apply_planning_scene not available")
        return False

    co = CollisionObject()
    co.id = object_id
    co.header.frame_id = frame_id
    co.operation = CollisionObject.ADD
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [float(size[0]), float(size[1]), float(size[2])]
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.w = 1.0
    co.primitives.append(box)
    co.primitive_poses.append(pose)

    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects.append(co)

    req = ApplyPlanningScene.Request()
    req.scene = scene
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    res = fut.result()
    ok = bool(res and res.success)
    node.get_logger().info(
        f"Planning obstacle '{object_id}' at {xyz} size={size} → {'OK' if ok else 'FAIL'}"
    )
    return ok


def clear_obstacle(
    node: Node,
    object_id: str = "table_clutter_box",
    frame_id: str = "base_link",
    timeout: float = 5.0,
) -> bool:
    cli = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    if not cli.wait_for_service(timeout_sec=timeout):
        return False
    co = CollisionObject()
    co.id = object_id
    co.header.frame_id = frame_id
    co.operation = CollisionObject.REMOVE
    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects.append(co)
    req = ApplyPlanningScene.Request()
    req.scene = scene
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    res = fut.result()
    return bool(res and res.success)
