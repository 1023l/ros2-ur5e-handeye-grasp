#!/usr/bin/env python3
"""把墙钟发布为 /clock，供假硬件 + MoveIt(use_sim_time:=true) 使用。"""

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


class WallClockPublisher(Node):
    def __init__(self):
        super().__init__('arm_wall_clock')
        self._pub = self.create_publisher(Clock, '/clock', 10)
        self.create_timer(0.01, self._tick)  # 100 Hz

    def _tick(self):
        msg = Clock()
        msg.clock = self.get_clock().now().to_msg()
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = WallClockPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
