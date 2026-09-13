"""Pose from detection: pinhole + apparent-size depth (known object size)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from arm_system.block_detector import BlockDetection

if TYPE_CHECKING:
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Header


@dataclass
class PoseEstimate:
    u: float
    v: float
    z_m: float
    x_m: float
    y_m: float
    method: str
    size_px: float


def robust_size_px(det: BlockDetection) -> Tuple[float, str]:
    """
    Apparent size under partial occlusion:
      - near-square → geometric mean (stable)
      - flat/tall (bar occlusion) → use the longer side (unoccluded axis)
    """
    w = max(float(det.w), 1.0)
    h = max(float(det.h), 1.0)
    aspect = max(w, h) / min(w, h)
    if aspect >= 1.35:
        return max(w, h), "apparent_long_side"
    return float((w * h) ** 0.5), "apparent_size"


def depth_from_bbox(
    det: BlockDetection,
    *,
    fx: float,
    object_size_m: float,
    z_fallback: float,
    z_min: float = 0.25,
    z_max: float = 0.85,
) -> Tuple[float, str]:
    """z ≈ fx * L / size_px. Falls back if bbox degenerate."""
    if object_size_m <= 0:
        return z_fallback, "fixed_z"
    size_px, size_method = robust_size_px(det)
    if size_px < 8.0:
        return z_fallback, "fixed_z"
    z = float(fx) * float(object_size_m) / size_px
    if z < z_min or z > z_max:
        return float(np_clip(z, z_min, z_max)), "size_clamped"
    return z, size_method


def np_clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def estimate_camera_pose(
    det: BlockDetection,
    *,
    fx: float,
    fy: float,
    cx0: float,
    cy0: float,
    object_size_m: float = 0.06,
    z_fallback: float = 0.45,
) -> PoseEstimate:
    z, method = depth_from_bbox(
        det, fx=fx, object_size_m=object_size_m, z_fallback=z_fallback
    )
    x = (det.cx - cx0) * z / fx
    y = (det.cy - cy0) * z / fy
    size_px, _ = robust_size_px(det)
    return PoseEstimate(
        u=det.cx,
        v=det.cy,
        z_m=z,
        x_m=float(x),
        y_m=float(y),
        method=method,
        size_px=float(size_px),
    )


def to_pose_stamped(est: PoseEstimate, header: "Header") -> "PoseStamped":
    from geometry_msgs.msg import PoseStamped

    msg = PoseStamped()
    msg.header = header
    msg.pose.position.x = est.x_m
    msg.pose.position.y = est.y_m
    msg.pose.position.z = est.z_m
    msg.pose.orientation.w = 1.0
    return msg


def pixel_error(est_u: float, est_v: float, gt_u: float, gt_v: float) -> float:
    return float(((est_u - gt_u) ** 2 + (est_v - gt_v) ** 2) ** 0.5)


def rank_detections_for_target(
    detections: list,
    target_label: str,
) -> Optional[BlockDetection]:
    """Prefer matching label, then confidence * area."""
    matched = [d for d in detections if d.label == target_label]
    pool = matched if matched else list(detections)
    if not pool:
        return None
    pool.sort(key=lambda d: d.confidence * (d.w * d.h), reverse=True)
    return pool[0]
