"""Colored block detector for arm perception demos (OpenCV HSV)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BlockDetection:
    label: str
    confidence: float
    cx: float
    cy: float
    x: int
    y: int
    w: int
    h: int


# HSV ranges in OpenCV (H: 0-179)
_HSV_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "red": [
        ((0, 80, 80), (10, 255, 255)),
        ((160, 80, 80), (179, 255, 255)),
    ],
    "green": [((40, 60, 60), (85, 255, 255))],
    "blue": [((95, 60, 60), (130, 255, 255))],
}

_BGR_ANNOTATE = {
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "blue": (255, 80, 0),
}


def detect_blocks(
    image_bgr: np.ndarray,
    colors: Optional[List[str]] = None,
    min_area: int = 400,
) -> List[BlockDetection]:
    """Return detections sorted by area (largest first).

    Horizontal occlusion bars split / flatten blobs. We:
      1) vertically-biased morphological CLOSE to bridge the bar gap
      2) keep small fragments then merge same-label neighbors
      3) use mask moments for center (more stable than bbox mid when fill is low)
    """
    if colors is None:
        colors = ["red", "green", "blue"]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    out: List[BlockDetection] = []
    h_img, w_img = image_bgr.shape[:2]
    frag_min = max(60, int(min_area * 0.25))

    for label in colors:
        ranges = _HSV_RANGES.get(label)
        if not ranges:
            continue
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Bridge horizontal occlusion bars: tall kernel (rows=height, cols=width)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < frag_min:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Moments on filled contour → center of mass (handles C-shape / bar hole)
            m = cv2.moments(cnt)
            if m["m00"] > 1e-3:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
            else:
                cx = x + w * 0.5
                cy = y + h * 0.5
            fill = area / max(float(w * h), 1.0)
            # Recover near-square bbox when CLOSE left a flat remnant
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect >= 1.45 and max(w, h) >= 24:
                side = int(max(w, h))
                x = int(np.clip(cx - side * 0.5, 0, w_img - 1))
                y = int(np.clip(cy - side * 0.5, 0, h_img - 1))
                w = int(min(side, w_img - x))
                h = int(min(side, h_img - y))
            size_score = min(1.0, area / (0.08 * h_img * w_img))
            conf = float(np.clip(0.45 * fill + 0.55 * size_score, 0.0, 0.99))
            out.append(BlockDetection(label, conf, cx, cy, x, y, w, h))

    # Prefer vertical merges (occlusion bars); allow larger Y gap
    out = merge_nearby_same_label(out, gap_px=36.0, gap_y_extra=22.0)
    # Drop tiny leftovers after merge
    out = [d for d in out if d.w * d.h >= min_area]
    out.sort(key=lambda d: d.w * d.h, reverse=True)
    return out


def merge_nearby_same_label(
    detections: List[BlockDetection],
    *,
    gap_px: float = 28.0,
    gap_y_extra: float = 0.0,
) -> List[BlockDetection]:
    """Union-find style merge for split contours of the same color."""
    if len(detections) <= 1:
        return list(detections)
    boxes = list(detections)
    changed = True
    while changed:
        changed = False
        n = len(boxes)
        used = [False] * n
        merged: List[BlockDetection] = []
        for i in range(n):
            if used[i]:
                continue
            cur = boxes[i]
            used[i] = True
            group = [cur]
            for j in range(i + 1, n):
                if used[j]:
                    continue
                o = boxes[j]
                if o.label != cur.label:
                    continue
                if any(
                    _boxes_near(g, o, gap_px, gap_y_extra=gap_y_extra) for g in group
                ):
                    used[j] = True
                    group.append(o)
                    changed = True
            merged.append(_union_detections(group))
        boxes = merged
    return boxes


def _boxes_near(
    a: BlockDetection, b: BlockDetection, gap: float, *, gap_y_extra: float = 0.0
) -> bool:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    gx, gy = gap, gap + gap_y_extra
    return not (
        ax2 + gx < b.x or bx2 + gx < a.x or ay2 + gy < b.y or by2 + gy < a.y
    )


def _union_detections(group: List[BlockDetection]) -> BlockDetection:
    if len(group) == 1:
        return group[0]
    x0 = min(d.x for d in group)
    y0 = min(d.y for d in group)
    x1 = max(d.x + d.w for d in group)
    y1 = max(d.y + d.h for d in group)
    w, h = x1 - x0, y1 - y0
    area_sum = float(sum(d.w * d.h for d in group))
    # Area-weighted center of mass across fragments
    cx = sum(d.cx * d.w * d.h for d in group) / max(area_sum, 1.0)
    cy = sum(d.cy * d.w * d.h for d in group) / max(area_sum, 1.0)
    conf = float(
        np.clip(
            sum(d.confidence * d.w * d.h for d in group) / max(area_sum, 1.0),
            0.0,
            0.99,
        )
    )
    return BlockDetection(
        label=group[0].label,
        confidence=conf,
        cx=float(cx),
        cy=float(cy),
        x=x0,
        y=y0,
        w=w,
        h=h,
    )


def annotate(image_bgr: np.ndarray, detections: List[BlockDetection]) -> np.ndarray:
    vis = image_bgr.copy()
    for d in detections:
        color = _BGR_ANNOTATE.get(d.label, (255, 255, 255))
        cv2.rectangle(vis, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        cv2.circle(vis, (int(d.cx), int(d.cy)), 4, color, -1)
        cv2.putText(
            vis,
            f"{d.label} {d.confidence:.2f}",
            (d.x, max(16, d.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis


def _table_background(width: int, height: int) -> np.ndarray:
    img = np.full((height, width, 3), 48, dtype=np.uint8)
    for y in range(height // 2, height):
        img[y, :] = (56 + (y - height // 2) // 8,) * 3
    return img


def make_synthetic_scene(
    width: int = 640,
    height: int = 480,
    *,
    color: str = "red",
    cx: float = 320.0,
    cy: float = 240.0,
    box: int = 70,
) -> np.ndarray:
    """Gray table + one solid color block (for no-camera demos)."""
    img = _table_background(width, height)
    bgr = _BGR_ANNOTATE.get(color, (0, 0, 255))
    x0 = int(np.clip(cx - box / 2, 0, width - 1))
    y0 = int(np.clip(cy - box / 2, 0, height - 1))
    x1 = int(np.clip(x0 + box, 0, width))
    y1 = int(np.clip(y0 + box, 0, height))
    img[y0:y1, x0:x1] = bgr
    noise = np.random.randint(0, 12, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def make_synthetic_photo_scene(
    width: int,
    height: int,
    patch_bgr: np.ndarray,
    *,
    cx: float,
    cy: float,
    patch_h: int = 140,
) -> np.ndarray:
    """Paste a real photo crop onto the table so COCO-YOLO can fire in sim."""
    img = _table_background(width, height)
    ph, pw = patch_bgr.shape[:2]
    if ph < 1 or pw < 1:
        return img
    scale = float(patch_h) / float(ph)
    nw = max(8, int(pw * scale))
    nh = max(8, int(ph * scale))
    patch = cv2.resize(patch_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = int(np.clip(cx - nw / 2, 0, width - 1))
    y0 = int(np.clip(cy - nh / 2, 0, height - 1))
    x1 = min(width, x0 + nw)
    y1 = min(height, y0 + nh)
    pw2, ph2 = x1 - x0, y1 - y0
    img[y0:y1, x0:x1] = patch[:ph2, :pw2]
    return img
