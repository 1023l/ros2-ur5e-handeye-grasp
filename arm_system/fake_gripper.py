"""Fake parallel gripper for demos (no real hardware)."""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker


class FakeGripper(Node):
    """
    /arm/gripper/set  (std_srvs/SetBool): data=True close, False open
    /arm/gripper/closed (Bool)
    /arm/gripper/state  (String): open|closed
    /arm/gripper/marker (Marker): green box when closed (visual cue)
    """

    def __init__(self) -> None:
        super().__init__("arm_fake_gripper")
        self._closed = False
        self._pub_closed = self.create_publisher(Bool, "/arm/gripper/closed", 10)
        self._pub_state = self.create_publisher(String, "/arm/gripper/state", 10)
        self._pub_marker = self.create_publisher(Marker, "/arm/gripper/marker", 10)
        self.create_service(SetBool, "/arm/gripper/set", self._on_set)
        self.create_timer(0.2, self._tick)
        self.get_logger().info("Fake gripper ready: service /arm/gripper/set")

    def _on_set(self, request: SetBool.Request, response: SetBool.Response):
        self._closed = bool(request.data)
        response.success = True
        response.message = "closed" if self._closed else "open"
        self.get_logger().info(f"Gripper → {response.message}")
        self._publish()
        return response

    def _tick(self) -> None:
        self._publish()

    def _publish(self) -> None:
        self._pub_closed.publish(Bool(data=self._closed))
        self._pub_state.publish(String(data="closed" if self._closed else "open"))

        m = Marker()
        m.header.frame_id = "tool0"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "fake_gripper"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD if self._closed else Marker.DELETE
        m.pose.position.z = 0.04
        m.pose.orientation.w = 1.0
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.scale.z = 0.05
        m.color.r = 0.2
        m.color.g = 0.9
        m.color.b = 0.2
        m.color.a = 0.85
        self._pub_marker.publish(m)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = FakeGripper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
