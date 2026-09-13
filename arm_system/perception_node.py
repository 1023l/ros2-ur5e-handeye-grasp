"""Arm perception node: image → detections → camera-frame pose.

Backends: color | yolo | auto
Scenes: easy (single block) | hard (clutter + occlusion + multi-target)
Depth: fixed Z or apparent-size (known object size)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from arm_system.block_detector import (
    BlockDetection,
    annotate,
    detect_blocks,
    make_synthetic_photo_scene,
    make_synthetic_scene,
)
from arm_system.grasp_ranking import rank_grasp_candidates
from arm_system.hard_scene import make_hard_scene, select_target_gt
from arm_system.pose_from_bbox import (
    estimate_camera_pose,
    to_pose_stamped,
)
from arm_system.yolo_detector import (
    YoloDetector,
    default_models_dir,
    parse_class_filter,
    resolve_yolo_weights,
)


class ArmPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_perception")
        self.declare_parameter("use_synthetic", True)
        self.declare_parameter("image_topic", "/arm/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/arm/camera/camera_info")
        self.declare_parameter("camera_frame", "arm_camera_optical_frame")
        self.declare_parameter("target_color", "red")
        self.declare_parameter("object_distance_m", 0.45)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("fx", 600.0)
        self.declare_parameter("fy", 600.0)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("detector_backend", "auto")
        self.declare_parameter("yolo_model_path", "")
        self.declare_parameter("yolo_conf", 0.35)
        self.declare_parameter("yolo_classes", "person,bottle,cup,sports ball,bus,car")
        self.declare_parameter("yolo_device", "cpu")
        self.declare_parameter("demo_patch_path", "")
        # Depth / hard scene
        self.declare_parameter("scene_mode", "hard")  # easy|hard
        self.declare_parameter("object_size_m", 0.06)
        self.declare_parameter("use_apparent_size", True)
        self.declare_parameter("occlusion_prob", 0.45)
        self.declare_parameter("n_distractors", 3)

        self._bridge = CvBridge()
        self._use_synthetic = bool(self.get_parameter("use_synthetic").value)
        self._image_topic = str(self.get_parameter("image_topic").value)
        self._info_topic = str(self.get_parameter("camera_info_topic").value)
        self._frame = str(self.get_parameter("camera_frame").value)
        self._color = str(self.get_parameter("target_color").value)
        self._z = float(self.get_parameter("object_distance_m").value)
        self._w = int(self.get_parameter("image_width").value)
        self._h = int(self.get_parameter("image_height").value)
        self._fx = float(self.get_parameter("fx").value)
        self._fy = float(self.get_parameter("fy").value)
        self._backend_req = str(self.get_parameter("detector_backend").value).lower()
        self._scene_mode = str(self.get_parameter("scene_mode").value).lower()
        self._object_size = float(self.get_parameter("object_size_m").value)
        self._use_apparent = bool(self.get_parameter("use_apparent_size").value)
        self._occlusion_prob = float(self.get_parameter("occlusion_prob").value)
        self._n_distractors = int(self.get_parameter("n_distractors").value)
        self._yolo_classes = parse_class_filter(
            str(self.get_parameter("yolo_classes").value)
        )
        self._yolo: Optional[YoloDetector] = None
        self._demo_patch: Optional[np.ndarray] = None
        self._active_backend = self._init_backend()
        self._last_gt = None

        self._cx0 = self._w * 0.5
        self._cy0 = self._h * 0.5
        self._t = 0.0

        self._pub_image = self.create_publisher(Image, self._image_topic, 10)
        self._pub_info = self.create_publisher(CameraInfo, self._info_topic, 10)
        self._pub_debug = self.create_publisher(Image, "/arm/debug_image", 10)
        self._pub_det = self.create_publisher(Detection2DArray, "/arm/detections", 10)
        self._pub_pose = self.create_publisher(PoseStamped, "/arm/target_pose", 10)
        self._pub_gt = self.create_publisher(String, "/arm/scene_gt", 10)

        if self._use_synthetic:
            hz = float(self.get_parameter("publish_hz").value)
            self.create_timer(1.0 / max(hz, 1.0), self._on_timer_synthetic)
            self.get_logger().info(
                f"Synthetic ON backend={self._active_backend} "
                f"scene={self._scene_mode} apparent_size={self._use_apparent}"
            )
        else:
            self.create_subscription(Image, self._image_topic, self._on_image, 10)
            self.get_logger().info(
                f"Subscribing {self._image_topic} backend={self._active_backend}"
            )

    def _init_backend(self) -> str:
        want = self._backend_req
        if want not in ("auto", "yolo", "color"):
            self.get_logger().warn(f"Unknown detector_backend={want}, using auto")
            want = "auto"
        if want == "color":
            return "color"

        weights = resolve_yolo_weights(str(self.get_parameter("yolo_model_path").value))
        if weights is None:
            weights = default_models_dir() / "yolov8n.pt"
            if not weights.is_file():
                weights = None

        if weights is None:
            if want == "yolo":
                self.get_logger().error(
                    "detector_backend=yolo but no weights. "
                    "Run: ros2 run arm_system setup_yolo_weights.py"
                )
            else:
                self.get_logger().info("No YOLO weights → color backend")
            return "color"

        try:
            self._yolo = YoloDetector(
                str(weights),
                conf=float(self.get_parameter("yolo_conf").value),
                class_filter=self._yolo_classes,
                device=str(self.get_parameter("yolo_device").value),
            )
            self._demo_patch = self._load_demo_patch()
            self.get_logger().info(f"YOLO ready: {weights}")
            return "yolo"
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"YOLO load failed ({exc}) → color")
            self._yolo = None
            return "color"

    def _load_demo_patch(self) -> Optional[np.ndarray]:
        explicit = str(self.get_parameter("demo_patch_path").value).strip()
        candidates = []
        if explicit:
            candidates.append(Path(explicit))
        candidates.append(default_models_dir() / "demo_target.jpg")
        for p in candidates:
            if p.is_file():
                img = cv2.imread(str(p))
                if img is not None:
                    return img
        return None

    def _on_timer_synthetic(self) -> None:
        self._t += 0.05
        self._last_gt = None
        if self._active_backend == "yolo" and self._demo_patch is not None:
            cx = self._w * 0.5 + 120.0 * math.sin(self._t)
            cy = self._h * 0.5 + 80.0 * math.cos(self._t * 0.7)
            img = make_synthetic_photo_scene(
                self._w, self._h, self._demo_patch, cx=cx, cy=cy, patch_h=150
            )
        elif self._scene_mode == "hard" and self._active_backend == "color":
            hard = make_hard_scene(
                self._w,
                self._h,
                target_label=self._color,
                t=self._t,
                fx=self._fx,
                object_size_m=self._object_size,
                n_distractors=self._n_distractors,
                occlusion_prob=self._occlusion_prob,
                seed=None,
            )
            img = hard.image_bgr
            self._last_gt = hard
            gt_obj = select_target_gt(hard, self._color)
            if gt_obj is not None:
                payload = {
                    "label": gt_obj.label,
                    "u": gt_obj.cx,
                    "v": gt_obj.cy,
                    "z": gt_obj.z_m,
                    "occluded": gt_obj.occluded,
                    "n_objects": len(hard.objects),
                }
                self._pub_gt.publish(String(data=json.dumps(payload)))
        else:
            cx = self._w * 0.5 + 120.0 * math.sin(self._t)
            cy = self._h * 0.5 + 80.0 * math.cos(self._t * 0.7)
            img = make_synthetic_scene(
                self._w, self._h, color=self._color, cx=cx, cy=cy, box=72
            )

        stamp = self.get_clock().now().to_msg()
        self._publish_camera_info(stamp)
        self._pub_image.publish(self._bridge.cv2_to_imgmsg(img, encoding="bgr8"))
        self._process(img, stamp)

    def _on_image(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"cv_bridge failed: {exc}")
            return
        self._process(img, msg.header.stamp)

    def _process(self, image_bgr: np.ndarray, stamp) -> None:
        if self._active_backend == "yolo" and self._yolo is not None:
            detections = self._yolo.detect(image_bgr)
        else:
            detections = detect_blocks(
                image_bgr, colors=["red", "green", "blue"], min_area=200
            )

        ranked = rank_grasp_candidates(
            detections,
            image_w=image_bgr.shape[1],
            image_h=image_bgr.shape[0],
            target_label=self._color,
            top_k=5,
        )
        best = ranked[0].det if ranked else None

        debug = annotate(image_bgr, detections)
        tag = f"{self._active_backend}/{self._scene_mode}"
        cv2.putText(
            debug, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2
        )
        if ranked:
            cv2.putText(
                debug,
                f"best={ranked[0].det.label} s={ranked[0].score:.2f}",
                (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            # mark top-3
            for i, sc in enumerate(ranked[:3]):
                d = sc.det
                cv2.putText(
                    debug,
                    f"#{i+1}",
                    (d.x, max(14, d.y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
        debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_msg.header.stamp = stamp
        debug_msg.header.frame_id = self._frame
        self._pub_debug.publish(debug_msg)

        det_arr = Detection2DArray()
        det_arr.header = Header(stamp=stamp, frame_id=self._frame)
        for d in detections:
            msg = Detection2D()
            msg.header = det_arr.header
            msg.bbox.center.position.x = d.cx
            msg.bbox.center.position.y = d.cy
            msg.bbox.size_x = float(d.w)
            msg.bbox.size_y = float(d.h)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = d.label
            hyp.hypothesis.score = float(d.confidence)
            msg.results.append(hyp)
            det_arr.detections.append(msg)
        self._pub_det.publish(det_arr)

        if best is None:
            return

        header = Header(stamp=stamp, frame_id=self._frame)
        size_m = self._object_size if self._use_apparent else 0.0
        est = estimate_camera_pose(
            best,
            fx=self._fx,
            fy=self._fy,
            cx0=self._cx0,
            cy0=self._cy0,
            object_size_m=size_m,
            z_fallback=self._z,
        )
        self._pub_pose.publish(to_pose_stamped(est, header))

    def _publish_camera_info(self, stamp) -> None:
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._frame
        info.width = self._w
        info.height = self._h
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [
            self._fx, 0.0, self._cx0,
            0.0, self._fy, self._cy0,
            0.0, 0.0, 1.0,
        ]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            self._fx, 0.0, self._cx0, 0.0,
            0.0, self._fy, self._cy0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self._pub_info.publish(info)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
