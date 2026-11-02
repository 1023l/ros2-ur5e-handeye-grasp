"""Train-time image augment (YOLO-style sim-to-real hygiene)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    brightness: float = 0.15
    contrast: float = 0.15
    gauss_noise_std: float = 0.04
    jpeg_quality_range: tuple = (70, 95)
    hflip_prob: float = 0.0  # off: breaks left/right AGV layout semantics
    random_crop_frac: float = 0.08


def augment_rgb(
    rgb01: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[AugmentConfig] = None,
) -> np.ndarray:
    """rgb01: float32 HWC [0,1] → augmented float32 [0,1]."""
    cfg = cfg or AugmentConfig()
    img = np.clip(rgb01, 0.0, 1.0).astype(np.float32)
    if cfg.random_crop_frac > 0 and rng.random() < 0.5:
        h, w = img.shape[:2]
        margin = int(min(h, w) * cfg.random_crop_frac)
        if margin >= 2:
            y0 = rng.integers(0, margin)
            x0 = rng.integers(0, margin)
            y1 = h - rng.integers(0, margin)
            x1 = w - rng.integers(0, margin)
            crop = img[y0:y1, x0:x1]
            img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    if cfg.brightness > 0:
        b = 1.0 + float(rng.uniform(-cfg.brightness, cfg.brightness))
        img = np.clip(img * b, 0.0, 1.0)

    if cfg.contrast > 0:
        c = 1.0 + float(rng.uniform(-cfg.contrast, cfg.contrast))
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * c + mean, 0.0, 1.0)

    if cfg.gauss_noise_std > 0:
        img = np.clip(
            img + rng.normal(0, cfg.gauss_noise_std, img.shape).astype(np.float32),
            0.0,
            1.0,
        )

    if cfg.jpeg_quality_range and rng.random() < 0.35:
        q = int(rng.integers(cfg.jpeg_quality_range[0], cfg.jpeg_quality_range[1] + 1))
        bgr = (img * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return img.astype(np.float32)
