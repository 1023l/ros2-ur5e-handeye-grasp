"""Synthetic eye-to-hand scene: fabric roll on table + docked AGV load bay.

Not a toy color-block demo — models the industrial story:
detect-finished roll → pick → place into AGV bay (vehicle already docked).
AGV drive / tip-dump are out of scope; only the arm transfer segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from arm_system.policy.constants import (
    AGV_BAY_HALF,
    AGV_BAY_XY,
    IMG_FULL_H,
    IMG_FULL_W,
    TABLE_Z,
)


@dataclass
class AgvSceneGT:
    image_bgr: np.ndarray
    roll_xyz: Tuple[float, float, float]  # base frame meters
    roll_yaw: float
    roll_uv: Tuple[float, float]  # pixel of roll center
    agv_bay_xy: Tuple[float, float]
    agv_bay_uv: Tuple[float, float]
    occluded: bool = False
    clutter_count: int = 0
    extras: Dict = field(default_factory=dict)


@dataclass
class ScenePerturb:
    """Eval / domain-randomization knobs (industrial dock + pose noise)."""

    roll_xy_noise: float = 0.0  # meters added to roll after "belief" formed
    roll_yaw_noise: float = 0.0
    agv_dock_xy_noise: float = 0.0  # bay shifts after plan (late dock error)
    perception_xy_noise: float = 0.0  # what geometric baseline believes
    occlusion: float = 0.0  # 0..1 probability of bar occlusion
    lighting_gain: float = 1.0
    clutter: int = 2


def _table_bg(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    base = np.full((h, w, 3), 42, dtype=np.uint8)
    # concrete-ish floor beyond table
    base[int(h * 0.62) :, :] = (55, 52, 48)
    # table region
    cv2.rectangle(base, (40, 40), (w - 40, int(h * 0.62)), (68, 72, 78), -1)
    # slight grain
    grain = rng.integers(0, 12, (h, w, 3), dtype=np.uint8)
    return cv2.add(base, grain)


def _world_to_uv(
    x: float,
    y: float,
    z: float,
    *,
    width: int = IMG_FULL_W,
    height: int = IMG_FULL_H,
) -> Tuple[float, float]:
    """Eye-to-hand table projection (orthographic-ish).

    Maps workspace x∈[0.18,0.63], y∈[-0.32,0.28] into the upper table region
    so both roll and AGV bay stay on-image (previous pinhole put bay off-frame).
    z only nudges scale slightly.
    """
    u = (x - 0.18) / 0.45 * (width - 80.0) + 40.0
    v = (0.28 - y) / 0.60 * (height * 0.52) + 55.0
    # nearer (smaller z) → slightly larger / lower
    v += (0.20 - float(z)) * 40.0
    return float(u), float(v)


def _draw_roll(
    img: np.ndarray,
    uv: Tuple[float, float],
    yaw: float,
    size_px: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    u, v = uv
    bw, bh = size_px
    box = ((u, v), (bw, bh), np.degrees(yaw))
    corners = cv2.boxPoints(box).astype(np.int32)
    cv2.fillConvexPoly(img, corners, color)
    # end-cap rings (fabric roll cue)
    for sign in (-1.0, 1.0):
        ex = u + sign * 0.45 * bw * np.cos(yaw)
        ey = v + sign * 0.45 * bw * np.sin(yaw)
        cv2.circle(img, (int(ex), int(ey)), max(3, bh // 3), (220, 220, 220), 1)
    cv2.polylines(img, [corners], True, (240, 240, 240), 1)


def _draw_agv_bay(
    img: np.ndarray,
    uv: Tuple[float, float],
    half_uv: Tuple[float, float],
) -> None:
    u, v = uv
    hx, hy = half_uv
    x0, y0 = int(u - hx), int(v - hy)
    x1, y1 = int(u + hx), int(v + hy)
    # AGV body
    cv2.rectangle(img, (x0 - 18, y0 - 10), (x1 + 18, y1 + 28), (35, 38, 90), -1)
    # load bay slot — saturated green for classical detector
    cv2.rectangle(img, (x0, y0), (x1, y1), (40, 180, 40), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (80, 255, 120), 2)
    cv2.putText(
        img,
        "AGV",
        (x0 - 10, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 255),
        1,
        cv2.LINE_AA,
    )


def make_agv_scene(
    *,
    seed: Optional[int] = None,
    roll_xy: Optional[Tuple[float, float]] = None,
    roll_yaw: float = 0.0,
    agv_bay_xy: Optional[Tuple[float, float]] = None,
    perturb: Optional[ScenePerturb] = None,
    width: int = IMG_FULL_W,
    height: int = IMG_FULL_H,
) -> AgvSceneGT:
    rng = np.random.default_rng(seed)
    p = perturb or ScenePerturb()
    img = _table_bg(width, height, rng)

    if roll_xy is None:
        rx = float(rng.uniform(0.28, 0.48))
        ry = float(rng.uniform(-0.05, 0.22))
    else:
        rx, ry = float(roll_xy[0]), float(roll_xy[1])
    rx += float(rng.normal(0, p.roll_xy_noise)) if p.roll_xy_noise else 0.0
    ry += float(rng.normal(0, p.roll_xy_noise)) if p.roll_xy_noise else 0.0
    yaw = float(roll_yaw + rng.normal(0, p.roll_yaw_noise))

    if agv_bay_xy is None:
        ax, ay = AGV_BAY_XY
    else:
        ax, ay = float(agv_bay_xy[0]), float(agv_bay_xy[1])
    ax += float(rng.normal(0, p.agv_dock_xy_noise)) if p.agv_dock_xy_noise else 0.0
    ay += float(rng.normal(0, p.agv_dock_xy_noise)) if p.agv_dock_xy_noise else 0.0

    rz = TABLE_Z + 0.04
    roll_uv = _world_to_uv(rx, ry, rz)
    bay_uv = _world_to_uv(ax, ay, TABLE_Z + 0.02)
    # bay apparent size in pixels (approx)
    half_uv = (55.0, 42.0)

    # clutter strips (other rolls / pallets)
    for _ in range(max(0, p.clutter)):
        cx = float(rng.uniform(80, width - 80))
        cy = float(rng.uniform(90, height * 0.55))
        color = (
            int(rng.integers(40, 90)),
            int(rng.integers(40, 90)),
            int(rng.integers(40, 100)),
        )
        _draw_roll(
            img,
            (cx, cy),
            float(rng.uniform(-0.8, 0.8)),
            (int(rng.integers(40, 70)), int(rng.integers(16, 26))),
            color,
        )

    _draw_agv_bay(img, bay_uv, half_uv)
    # target roll — dyed fabric tone
    fabric = (int(40 + 30 * rng.random()), int(60 + 40 * rng.random()), int(140 + 60 * rng.random()))
    _draw_roll(img, roll_uv, yaw, (72, 28), fabric)

    occluded = False
    if rng.random() < p.occlusion:
        u, v = roll_uv
        cv2.rectangle(
            img,
            (int(u - 50), int(v - 8)),
            (int(u + 50), int(v + 10)),
            (25, 25, 25),
            -1,
        )
        occluded = True

    if abs(p.lighting_gain - 1.0) > 1e-3:
        img = np.clip(img.astype(np.float32) * p.lighting_gain, 0, 255).astype(np.uint8)

    noise = rng.integers(0, 16, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return AgvSceneGT(
        image_bgr=img,
        roll_xyz=(rx, ry, rz),
        roll_yaw=yaw,
        roll_uv=roll_uv,
        agv_bay_xy=(ax, ay),
        agv_bay_uv=bay_uv,
        occluded=occluded,
        clutter_count=p.clutter,
        extras={"perception_xy_noise": p.perception_xy_noise},
    )


def in_agv_bay(
    xy: Tuple[float, float],
    bay_xy: Tuple[float, float],
    half: Tuple[float, float] = AGV_BAY_HALF,
) -> bool:
    return abs(xy[0] - bay_xy[0]) <= half[0] and abs(xy[1] - bay_xy[1]) <= half[1]


def perceive_roll(
    gt: AgvSceneGT,
    *,
    rng: Optional[np.random.Generator] = None,
    perception_xy_noise: float = 0.0,
) -> Tuple[float, float]:
    """Geometric baseline 'vision' — GT plus isotropic noise (no magic GT leak)."""
    rng = rng or np.random.default_rng()
    nx = float(rng.normal(0, perception_xy_noise)) if perception_xy_noise else 0.0
    ny = float(rng.normal(0, perception_xy_noise)) if perception_xy_noise else 0.0
    return gt.roll_xyz[0] + nx, gt.roll_xyz[1] + ny


def estimate_bay_xy_from_image(
    image_bgr: np.ndarray,
    *,
    fallback_xy: Tuple[float, float] = AGV_BAY_XY,
) -> Tuple[float, float, float]:
    """Detect green AGV load-slot → base XY via grid search on the same pinhole.

    Returns (x, y, confidence in [0,1]). Classical vision — complements ACT.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 50, 50), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return float(fallback_xy[0]), float(fallback_xy[1]), 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 200:
        return float(fallback_xy[0]), float(fallback_xy[1]), 0.0
    m = cv2.moments(c)
    if abs(m["m00"]) < 1e-6:
        return float(fallback_xy[0]), float(fallback_xy[1]), 0.0
    u = m["m10"] / m["m00"]
    v = m["m01"] / m["m00"]
    z = TABLE_Z + 0.02
    best = None
    best_err = 1e9
    for x in np.linspace(0.25, 0.60, 36):
        for y in np.linspace(-0.35, 0.10, 36):
            pu, pv = _world_to_uv(float(x), float(y), z)
            err = (pu - u) ** 2 + (pv - v) ** 2
            if err < best_err:
                best_err = err
                best = (float(x), float(y))
    if best is None or best_err > 40 ** 2:
        return float(fallback_xy[0]), float(fallback_xy[1]), 0.0
    conf = float(np.clip(area / 3500.0, 0.0, 1.0))
    conf *= float(np.clip(1.0 - best_err / (40 ** 2), 0.0, 1.0))
    return best[0], best[1], conf


EVAL_PROFILES: Dict[str, ScenePerturb] = {
    "nominal": ScenePerturb(
        roll_xy_noise=0.005,
        agv_dock_xy_noise=0.005,
        perception_xy_noise=0.008,
        occlusion=0.1,
        clutter=2,
    ),
    "agv_dock_perturb": ScenePerturb(
        roll_xy_noise=0.012,
        agv_dock_xy_noise=0.055,  # late dock shift beyond tight bay half-size
        perception_xy_noise=0.015,
        occlusion=0.15,
        clutter=3,
        lighting_gain=0.92,
    ),
    "clutter_occlusion": ScenePerturb(
        roll_xy_noise=0.01,
        agv_dock_xy_noise=0.015,
        perception_xy_noise=0.02,
        occlusion=0.65,
        clutter=5,
        lighting_gain=0.85,
    ),
}
