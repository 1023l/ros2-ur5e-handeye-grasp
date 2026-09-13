"""Hard synthetic scenes: multi-target, clutter, partial occlusion + GT labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from arm_system.block_detector import _BGR_ANNOTATE, _table_background


@dataclass
class SceneObjectGT:
    label: str
    cx: float
    cy: float
    z_m: float
    box_px: int
    occluded: bool = False


@dataclass
class HardSceneResult:
    image_bgr: np.ndarray
    objects: List[SceneObjectGT] = field(default_factory=list)
    target_label: str = "red"


def make_hard_scene(
    width: int = 640,
    height: int = 480,
    *,
    target_label: str = "red",
    seed: Optional[int] = None,
    t: float = 0.0,
    fx: float = 600.0,
    object_size_m: float = 0.06,
    n_distractors: int = 3,
    occlusion_prob: float = 0.55,
    competing_same_label: bool = False,
) -> HardSceneResult:
    """
    Build a cluttered table top.

    True depth z varies per object; bbox size follows pinhole: box_px ≈ fx * size / z.
    Distractor colors + gray junk; optional horizontal bar partially covering target.
    """
    rng = np.random.default_rng(seed)
    img = _table_background(width, height)
    objects: List[SceneObjectGT] = []

    # Place target first (orbit slowly so tracking isn't trivial freeze-frame)
    tx = width * 0.5 + 90.0 * np.sin(t)
    ty = height * 0.45 + 50.0 * np.cos(t * 0.8)
    tz = float(rng.uniform(0.38, 0.55))
    tbox = int(np.clip(fx * object_size_m / tz, 28, 140))
    _draw_block(img, target_label, tx, ty, tbox)
    objects.append(
        SceneObjectGT(target_label, float(tx), float(ty), tz, tbox, occluded=False)
    )

    colors = [c for c in ("red", "green", "blue") if c != target_label]
    for i in range(n_distractors):
        label = colors[i % len(colors)]
        attempts = 0
        while attempts < 20:
            cx = float(rng.uniform(80, width - 80))
            cy = float(rng.uniform(120, height - 80))
            if abs(cx - tx) + abs(cy - ty) > 90:
                break
            attempts += 1
        z = float(rng.uniform(0.35, 0.60))
        box = int(np.clip(fx * object_size_m / z, 24, 130))
        _draw_block(img, label, cx, cy, box)
        objects.append(SceneObjectGT(label, cx, cy, z, box, False))

    # Same-label competitor near image edge (high uncertainty) for ranking ablations
    if competing_same_label:
        cx = float(rng.uniform(40, 100))
        cy = float(rng.uniform(80, 140))
        z = float(rng.uniform(0.42, 0.58))
        box = int(np.clip(fx * object_size_m / z, 24, 100))
        _draw_block(img, target_label, cx, cy, box)
        objects.append(SceneObjectGT(target_label, cx, cy, z, box, False))

    # Gray clutter junk
    for _ in range(4):
        x0 = int(rng.integers(20, width - 100))
        y0 = int(rng.integers(height // 2, height - 40))
        w = int(rng.integers(30, 90))
        h = int(rng.integers(18, 50))
        shade = int(rng.integers(70, 140))
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (shade, shade, shade), -1)

    # Partial occlusion over target
    if rng.random() < occlusion_prob:
        bar_h = max(8, tbox // 3)
        y0 = int(np.clip(ty - bar_h // 2, 0, height - bar_h))
        x0 = int(np.clip(tx - tbox * 0.7, 0, width - 1))
        x1 = int(np.clip(tx + tbox * 0.7, 0, width))
        cv2.rectangle(img, (x0, y0), (x1, y0 + bar_h), (30, 30, 30), -1)
        objects[0].occluded = True

    noise = rng.integers(0, 18, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return HardSceneResult(image_bgr=img, objects=objects, target_label=target_label)


def _draw_block(img: np.ndarray, label: str, cx: float, cy: float, box: int) -> None:
    h, w = img.shape[:2]
    bgr = _BGR_ANNOTATE.get(label, (0, 0, 255))
    x0 = int(np.clip(cx - box / 2, 0, w - 1))
    y0 = int(np.clip(cy - box / 2, 0, h - 1))
    x1 = int(np.clip(x0 + box, 0, w))
    y1 = int(np.clip(y0 + box, 0, h))
    img[y0:y1, x0:x1] = bgr
    # edge highlight so detector has structure under noise
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)


def select_target_gt(
    scene: HardSceneResult, label: Optional[str] = None
) -> Optional[SceneObjectGT]:
    want = label or scene.target_label
    for o in scene.objects:
        if o.label == want:
            return o
    return None
