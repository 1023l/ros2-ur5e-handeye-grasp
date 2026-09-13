#!/usr/bin/env python3
"""One-shot NL command to the arm agent (or run skill locally if agent down).

Usage:
  ros2 run arm_system say.py "回家"
  ros2 run arm_system say.py "抓放"
  ros2 run arm_system say.py --local "帮助"
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from arm_system.intent import parse_intent
from arm_system.skills import SkillExecutor


class _StatusWaiter(Node):
    def __init__(self) -> None:
        super().__init__("arm_say_waiter")
        self.last: str | None = None
        self.create_subscription(String, "/arm/agent_status", self._cb, 10)

    def _cb(self, msg: String) -> None:
        self.last = msg.data


def publish_to_agent(text: str, wait_sec: float = 120.0) -> int:
    rclpy.init()
    node = _StatusWaiter()
    pub = node.create_publisher(String, "/arm/nl_command", 10)
    # latched-ish: wait for matching
    time.sleep(0.3)
    before = node.last
    pub.publish(String(data=text))
    node.get_logger().info(f"Published /arm/nl_command: {text}")
    deadline = time.time() + wait_sec
    saw_run = False
    code = 0
    while time.time() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.last is None or node.last == before:
            continue
        print(node.last)
        if node.last.startswith("run "):
            saw_run = True
        if node.last.startswith("OK ") or node.last.startswith("FAIL "):
            code = 0 if node.last.startswith("OK ") else 1
            break
        if node.last.startswith("unknown ") or node.last.startswith("busy "):
            code = 2
            break
    else:
        if not saw_run:
            print(
                "No agent response. Is arm_agent running?\n"
                "  ros2 run arm_system arm_agent.py\n"
                "Or: ros2 run arm_system say.py --local \"帮助\""
            )
            code = 2
    node.destroy_node()
    rclpy.shutdown()
    return code


def run_local(text: str) -> int:
    rclpy.init()
    skill, reason = parse_intent(text)
    print(f"intent: {skill} ({reason})")
    if skill is None:
        rclpy.shutdown()
        return 2
    ex = SkillExecutor()
    try:
        code, detail = ex.run(skill)
        print(detail)
        return code
    finally:
        ex.destroy()
        rclpy.shutdown()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Send NL command to arm agent")
    p.add_argument("text", nargs="+", help="Command text, e.g. 回家 / pick_place")
    p.add_argument(
        "--local",
        action="store_true",
        help="Run skill in-process (no arm_agent node)",
    )
    args = p.parse_args(argv)
    text = " ".join(args.text)
    if args.local:
        return run_local(text)
    return publish_to_agent(text)


if __name__ == "__main__":
    sys.exit(main())
