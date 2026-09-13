#!/usr/bin/env python3
"""Arm agent: natural language → skills.

Subscribe: /arm/nl_command (std_msgs/String)
Publish:   /arm/agent_status (std_msgs/String)

Examples:
  ros2 topic pub --once /arm/nl_command std_msgs/String "{data: '回家'}"
  ros2 run arm_system say.py "抓放"
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from arm_system.intent import parse_intent
from arm_system.skills import SkillExecutor


class ArmAgent(Node):
    def __init__(self) -> None:
        super().__init__("arm_agent")
        self.declare_parameter("use_llm", False)
        self.declare_parameter("llm_api_base", "https://api.openai.com/v1")
        self.declare_parameter("llm_api_key", "")
        self.declare_parameter("llm_model", "gpt-4o-mini")

        self._busy = False
        self._lock = threading.Lock()
        self._exec = SkillExecutor(logger=self.get_logger())
        self._status_pub = self.create_publisher(String, "/arm/agent_status", 10)
        self.create_subscription(String, "/arm/nl_command", self._on_cmd, 10)
        self._publish_status("idle — send Chinese/English to /arm/nl_command")
        self.get_logger().info(
            "Arm agent ready. Examples: 「回家」「抓放」「靠近」「帮助」"
        )

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _on_cmd(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text:
            return
        with self._lock:
            if self._busy:
                self._publish_status(f"busy — ignored: {text}")
                return
            self._busy = True
        # Run outside subscription callback to avoid nested rclpy.spin
        threading.Thread(target=self._handle_safe, args=(text,), daemon=True).start()

    def _handle_safe(self, text: str) -> None:
        try:
            self._handle(text)
        finally:
            with self._lock:
                self._busy = False

    def _handle(self, text: str) -> None:
        use_llm = bool(self.get_parameter("use_llm").value)
        skill, reason = parse_intent(
            text,
            use_llm=use_llm,
            api_base=str(self.get_parameter("llm_api_base").value),
            api_key=str(self.get_parameter("llm_api_key").value),
            model=str(self.get_parameter("llm_model").value),
        )
        if skill is None:
            self._publish_status(f"unknown intent ({reason}): {text}")
            return
        self._publish_status(f"run {skill} ({reason}) ← {text}")
        code, detail = self._exec.run(skill)
        level = "OK" if code == 0 else "FAIL"
        self._publish_status(f"{level} [{skill}] {detail}")

    def destroy_node(self) -> bool:
        self._exec.destroy()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
