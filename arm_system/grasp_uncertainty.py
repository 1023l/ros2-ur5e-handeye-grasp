"""Uncertainty-aware grasp scoring (risk-adjusted, not just mean confidence).

Senior-shaped difference vs heuristic ranking:
  score_risk = score_mean * exp(-λ * u)
where u aggregates depth / occlusion / edge / aspect uncertainty proxies.
Interview line: "we rank by risk-adjusted utility, not raw detector score."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from arm_system.block_detector import BlockDetection
from arm_system.grasp_ranking import ScoredGrasp, score_detection
from arm_system.pose_from_bbox import robust_size_px


@dataclass
class UncertainGrasp:
    scored: ScoredGrasp
    uncertainty: float
    risk_adjusted_score: float
    u_parts: dict


def estimate_grasp_uncertainty(
    det: BlockDetection,
    *,
    image_w: int,
    image_h: int,
    others: Sequence[BlockDetection] = (),
    fx: float = 600.0,
    object_size_m: float = 0.06,
) -> tuple[float, dict]:
    """
    Unitless uncertainty in ~[0, 1+]. Drivers:
      - bbox aspect (occlusion / truncation)
      - edge proximity
      - IoU with others
      - apparent-size depth sensitivity (dz/d(size_px))
      - low fill proxy via confidence
    """
    nx = abs(det.cx - image_w * 0.5) / max(image_w * 0.5, 1.0)
    ny = abs(det.cy - image_h * 0.5) / max(image_h * 0.5, 1.0)
    u_peripheral = float(np.clip(max(nx, ny), 0.0, 1.0))

    margin = 12.0
    near_edge = (
        det.x < margin
        or det.y < margin
        or det.x + det.w > image_w - margin
        or det.y + det.h > image_h - margin
    )
    u_edge = 0.7 if near_edge else 0.15 * u_peripheral

    max_iou = 0.0
    for o in others:
        if o is det:
            continue
        ax2, ay2 = det.x + det.w, det.y + det.h
        bx2, by2 = o.x + o.w, o.y + o.h
        ix1, iy1 = max(det.x, o.x), max(det.y, o.y)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter > 0:
            union = float(det.w * det.h + o.w * o.h - inter)
            max_iou = max(max_iou, inter / max(union, 1.0))
    u_iou = float(np.clip(max_iou * 1.2, 0.0, 1.0))

    size_px, _ = robust_size_px(det)
    u_depth = float(np.clip(3.0 / max(size_px, 1.0), 0.0, 1.0))
    if object_size_m > 0 and fx > 0:
        z = fx * object_size_m / max(size_px, 1.0)
        u_depth = float(np.clip(abs(z) * (1.0 / max(size_px, 1.0)) / 0.08, 0.0, 1.5))

    w, h = max(det.w, 1), max(det.h, 1)
    aspect = max(w, h) / min(w, h)
    u_aspect = float(np.clip((aspect - 1.0) / 1.5, 0.0, 1.0))
    u_conf = float(np.clip(1.0 - det.confidence, 0.0, 1.0))

    parts = {
        "u_aspect": u_aspect,
        "u_edge": u_edge,
        "u_iou": u_iou,
        "u_depth": u_depth,
        "u_conf": u_conf,
        "u_peripheral": u_peripheral,
    }
    # Workspace policy: peripheral / edge risk dominates (don't grasp margin junk)
    u = (
        0.28 * u_peripheral
        + 0.22 * u_edge
        + 0.20 * u_depth
        + 0.15 * u_aspect
        + 0.10 * u_iou
        + 0.05 * u_conf
    )
    return float(u), parts


def score_detection_risk_adjusted(
    det: BlockDetection,
    *,
    image_w: int,
    image_h: int,
    target_label: Optional[str] = None,
    others: Sequence[BlockDetection] = (),
    fx: float = 600.0,
    object_size_m: float = 0.06,
    lambda_risk: float = 1.25,
) -> UncertainGrasp:
    base = score_detection(
        det,
        image_w=image_w,
        image_h=image_h,
        target_label=target_label,
        others=others,
    )
    u, parts = estimate_grasp_uncertainty(
        det,
        image_w=image_w,
        image_h=image_h,
        others=others,
        fx=fx,
        object_size_m=object_size_m,
    )
    risk_score = float(base.score * np.exp(-lambda_risk * u))
    reasons = dict(base.reasons)
    reasons.update(parts)
    reasons["uncertainty"] = u
    reasons["risk_adjusted"] = risk_score
    return UncertainGrasp(
        scored=ScoredGrasp(det=det, score=base.score, reasons=reasons),
        uncertainty=u,
        risk_adjusted_score=risk_score,
        u_parts=parts,
    )


def rank_grasp_candidates_uncertain(
    detections: Sequence[BlockDetection],
    *,
    image_w: int,
    image_h: int,
    target_label: Optional[str] = None,
    top_k: int = 5,
    fx: float = 600.0,
    object_size_m: float = 0.06,
    lambda_risk: float = 1.25,
) -> list[UncertainGrasp]:
    ranked = [
        score_detection_risk_adjusted(
            d,
            image_w=image_w,
            image_h=image_h,
            target_label=target_label,
            others=detections,
            fx=fx,
            object_size_m=object_size_m,
            lambda_risk=lambda_risk,
        )
        for d in detections
    ]
    ranked.sort(key=lambda g: g.risk_adjusted_score, reverse=True)
    if target_label:
        matched = [g for g in ranked if g.scored.det.label == target_label]
        others = [g for g in ranked if g.scored.det.label != target_label]
        ranked = matched + others
    return ranked[: max(1, top_k)]
