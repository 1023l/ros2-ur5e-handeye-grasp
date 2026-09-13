"""Hand-eye TF bridge: load calibrated YAML (or demo defaults) and remap poses."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from arm_system.handeye_math import load_handeye_yaml
from arm_system.pose_utils import quat_from_rpy


def _quat_from_rpy(roll: float, pitch: float, yaw: float):
    q = quat_from_rpy(roll, pitch, yaw)
    return (q.x, q.y, q.z, q.w)


class HandEyeBridge(Node):
    """
    Publishes TF base → camera and /arm/target_pose_base.

    Prefer config/handeye_result.yaml from calibration; fall back to demo defaults.
    """

    def __init__(self) -> None:
        super().__init__("arm_handeye")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "arm_camera_optical_frame")
        self.declare_parameter("handeye_yaml", "")
        # Demo defaults (same as original static TF) if YAML missing.
        self.declare_parameter("cam_x", 0.40)
        self.declare_parameter("cam_y", 0.0)
        self.declare_parameter("cam_z", 0.50)
        self.declare_parameter("cam_roll", math.pi)
        self.declare_parameter("cam_pitch", 0.0)
        self.declare_parameter("cam_yaw", 0.0)

        self._base = str(self.get_parameter("base_frame").value)
        self._cam = str(self.get_parameter("camera_frame").value)
        self._xyz, self._q, src = self._load_extrinsic()
        self._broadcaster = TransformBroadcaster(self)
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)

        self.create_timer(0.05, self._broadcast_tf)
        self.create_subscription(PoseStamped, "/arm/target_pose", self._on_pose, 10)
        self._pub = self.create_publisher(PoseStamped, "/arm/target_pose_base", 10)

        self.get_logger().info(
            f"Hand-eye TF {self._base} → {self._cam} from [{src}] "
            f"t={self._xyz} Output: /arm/target_pose_base"
        )

    def _load_extrinsic(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float], str]:
        yaml_param = str(self.get_parameter("handeye_yaml").value).strip()
        candidates = []
        if yaml_param:
            candidates.append(Path(yaml_param))
        try:
            share = Path(get_package_share_directory("arm_system"))
            candidates.append(share / "config" / "handeye_result.yaml")
        except Exception:  # noqa: BLE001
            pass
        # Source-tree fallback when running via symlink-install mid-edit
        candidates.append(
            Path(__file__).resolve().parents[1] / "config" / "handeye_result.yaml"
        )

        for path in candidates:
            if path.is_file():
                try:
                    res = load_handeye_yaml(path)
                    self._base = res.frame_id or self._base
                    self._cam = res.child_frame_id or self._cam
                    return res.translation, res.rotation_xyzw, str(path)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"Failed to load {path}: {exc}")

        xyz = (
            float(self.get_parameter("cam_x").value),
            float(self.get_parameter("cam_y").value),
            float(self.get_parameter("cam_z").value),
        )
        q = _quat_from_rpy(
            float(self.get_parameter("cam_roll").value),
            float(self.get_parameter("cam_pitch").value),
            float(self.get_parameter("cam_yaw").value),
        )
        return xyz, q, "demo-defaults"

    def _broadcast_tf(self) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._base
        t.child_frame_id = self._cam
        t.transform.translation.x = self._xyz[0]
        t.transform.translation.y = self._xyz[1]
        t.transform.translation.z = self._xyz[2]
        t.transform.rotation.x = self._q[0]
        t.transform.rotation.y = self._q[1]
        t.transform.rotation.z = self._q[2]
        t.transform.rotation.w = self._q[3]
        self._broadcaster.sendTransform(t)

    def _on_pose(self, msg: PoseStamped) -> None:
        try:
            tf = self._buffer.lookup_transform(
                self._base,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"TF lookup failed: {exc}", throttle_duration_sec=2.0)
            return
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._base
        out.pose = do_transform_pose(msg.pose, tf)
        self._pub.publish(out)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = HandEyeBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
