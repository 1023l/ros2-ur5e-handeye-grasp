#!/usr/bin/env python3
"""ROS2: image → corrected AGV place XY (for MoveIt pick_place).

Usage (WSL Ubuntu-22.04 + Humble):
  ros2 launch arm_system ur5e_moveit_demo.launch.py launch_rviz:=false
  ros2 launch arm_system arm_vision_demo.launch.py
  ros2 run arm_system policy_dock_node.py --ckpt /path/to/act_ckpt

Then call service /arm/policy/correct_place with empty request → PlaceXY response.
Wire pick_place to use returned place_xy instead of fixed constant.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from arm_system.policy.constants import AGV_BAY_XY
from arm_system.policy.dock_controller import fuse_dock_bay
from arm_system.policy.geometry_teacher import EEState, _aa_from_tool_down, state_vector
from arm_system.policy.scene_agv import estimate_bay_xy_from_image

try:
    from arm_system.policy.train_loop import load_policy
except Exception:  # noqa: BLE001
    load_policy = None  # type: ignore

from std_srvs.srv import Trigger


class PolicyDockNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_dock_node")
        self.declare_parameter("ckpt_dir", "")
        self.declare_parameter("image_topic", "/arm/camera/image_raw")
        self.declare_parameter("dock_mode", "fusion")
        self._bridge = CvBridge()
        self._last_bgr: Optional[np.ndarray] = None
        self._ckpt = Path(self.get_parameter("ckpt_dir").get_parameter_value().string_value)
        topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self._dock_mode = self.get_parameter("dock_mode").get_parameter_value().string_value
        self.create_subscription(Image, topic, self._on_image, 10)
        self._srv = self.create_service(Trigger, "/arm/policy/correct_place", self._on_correct)
        self._model = self._stats = self._device = None
        if self._ckpt.is_dir() and load_policy is not None:
            try:
                self._model, self._stats, _, self._device = load_policy(self._ckpt)
                self.get_logger().info(f"Loaded policy from {self._ckpt}")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Policy load failed: {e} — vision-only fallback")
        else:
            self.get_logger().warn("No ckpt — vision-only dock correction")

    def _on_image(self, msg: Image) -> None:
        try:
            self._last_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:  # noqa: BLE001
            pass

    def _correct(self) -> Tuple[float, float, str]:
        bay_belief = AGV_BAY_XY
        if self._last_bgr is None:
            return bay_belief[0], bay_belief[1], "no_image"
        vis_x, vis_y, vis_conf = estimate_bay_xy_from_image(
            self._last_bgr, fallback_xy=bay_belief
        )
        learn_bay = bay_belief
        if self._model is not None and self._stats is not None:
            import cv2
            import torch

            from arm_system.policy.constants import IMG_H, IMG_W

            ee = EEState(
                xyz=np.array([0.3, 0.0, 0.35]),
                aa=_aa_from_tool_down(),
                gripper=1.0,
            )
            st = state_vector(ee, bay_belief, bay_belief)
            st_n = self._stats.norm_state(st).astype(np.float32)
            img = cv2.resize(self._last_bgr, (IMG_W, IMG_H))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
            state_t = torch.from_numpy(st_n).unsqueeze(0).to(self._device)
            with torch.no_grad():
                off = self._model.predict_dock_offset(img_t, state_t)[0].cpu().numpy()
            learn_bay = (
                bay_belief[0] + float(np.clip(off[0], -0.10, 0.10)),
                bay_belief[1] + float(np.clip(off[1], -0.10, 0.10)),
            )
        corrected, _, _ = fuse_dock_bay(
            bay_belief,
            vis_xy=(vis_x, vis_y),
            vis_conf=vis_conf,
            learn_xy=learn_bay,
            mode=self._dock_mode,  # type: ignore[arg-type]
        )
        return corrected[0], corrected[1], "ok"

    def _on_correct(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        x, y, tag = self._correct()
        res.success = tag == "ok"
        res.message = f"place_xy=({x:.4f},{y:.4f}) mode={self._dock_mode} tag={tag}"
        self.get_logger().info(res.message)
        return res


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = PolicyDockNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
