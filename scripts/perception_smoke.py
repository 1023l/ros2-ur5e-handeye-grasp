#!/usr/bin/env python3
"""Print /arm/target_pose a few times — quick smoke check without RViz image view."""

from __future__ import annotations

import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class Once(Node):
    def __init__(self, count: int = 5) -> None:
        super().__init__("arm_perception_smoke")
        self._left = count
        self.create_subscription(PoseStamped, "/arm/target_pose", self._cb, 10)
        self.get_logger().info("Waiting for /arm/target_pose ...")

    def _cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.get_logger().info(
            f"pose frame={msg.header.frame_id} "
            f"xyz=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
        )
        self._left -= 1
        if self._left <= 0:
            raise SystemExit(0)


def main() -> int:
    rclpy.init()
    node = Once(5)
    try:
        rclpy.spin(node)
    except SystemExit:
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 1


if __name__ == "__main__":
    sys.exit(main())
