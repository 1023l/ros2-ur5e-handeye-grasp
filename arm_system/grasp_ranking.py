"""Grasp candidate scoring and ranked selection (multi-hypothesis)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from arm_system.block_detector import BlockDetection


@dataclass
class ScoredGrasp:
    det: BlockDetection
    score: float
    reasons: dict


def score_detection(
    det: BlockDetection,
    *,
    image_w: int,
    image_h: int,
    target_label: Optional[str] = None,
    others: Sequence[BlockDetection] = (),
) -> ScoredGrasp:
    """
    Higher is better.
    Factors: class match, confidence, area, center preference, occlusion proxy
    (overlap with other boxes / edge proximity).
    """
    area = float(max(det.w * det.h, 1))
    area_n = min(1.0, area / (0.12 * image_w * image_h))
    conf = float(np.clip(det.confidence, 0.0, 1.0))
    label_bonus = 1.0 if (target_label is None or det.label == target_label) else 0.15

    # Prefer more central targets (workspace / less distorted)
    nx = abs(det.cx - image_w * 0.5) / max(image_w * 0.5, 1.0)
    ny = abs(det.cy - image_h * 0.5) / max(image_h * 0.5, 1.0)
    center = float(np.clip(1.0 - 0.55 * max(nx, ny), 0.05, 1.0))

    # Edge penalty (truncated boxes → bad size depth)
    margin = 12.0
    near_edge = (
        det.x < margin
        or det.y < margin
        or det.x + det.w > image_w - margin
        or det.y + det.h > image_h - margin
    )
    edge = 0.55 if near_edge else 1.0

    # Overlap with other detections as occlusion proxy
    occ = 1.0
    max_iou = 0.0
    for o in others:
        if o is det:
            continue
        iou = _iou(det, o)
        max_iou = max(max_iou, iou)
    if max_iou > 0.05:
        occ = float(np.clip(1.0 - 0.85 * max_iou, 0.15, 1.0))

    # Aspect: under horizontal bar occlusion, bbox becomes flat — still graspable
    # if long side is large; penalize ultra-thin junk.
    aspect = max(det.w, det.h) / max(min(det.w, det.h), 1)
    aspect_ok = float(np.clip(1.15 - 0.15 * max(0.0, aspect - 1.0), 0.35, 1.0))

    score = (
        label_bonus
        * (0.35 * conf + 0.28 * area_n + 0.20 * center + 0.17 * aspect_ok)
        * edge
        * occ
    )
    return ScoredGrasp(
        det=det,
        score=float(score),
        reasons={
            "conf": conf,
            "area_n": area_n,
            "center": center,
            "edge": edge,
            "occ": occ,
            "label_bonus": label_bonus,
            "iou": max_iou,
            "aspect_ok": aspect_ok,
        },
    )


def rank_grasp_candidates(
    detections: Sequence[BlockDetection],
    *,
    image_w: int,
    image_h: int,
    target_label: Optional[str] = None,
    top_k: int = 5,
) -> List[ScoredGrasp]:
    scored = [
        score_detection(d, image_w=image_w, image_h=image_h, target_label=target_label, others=detections)
        for d in detections
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    # Prefer target-label first among close scores
    if target_label:
        matched = [s for s in scored if s.det.label == target_label]
        others = [s for s in scored if s.det.label != target_label]
        scored = matched + others
    return scored[: max(1, top_k)]


def _iou(a: BlockDetection, b: BlockDetection) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    union = float(a.w * a.h + b.w * b.h - inter)
    return inter / max(union, 1.0)
