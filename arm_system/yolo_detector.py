"""Ultralytics YOLO wrapper for arm perception."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np

from arm_system.block_detector import BlockDetection


class YoloDetector:
    """Thin wrapper: BGR image → BlockDetection list."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        class_filter: Optional[Sequence[str]] = None,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(model_path)
        self._conf = float(conf)
        self._device = device
        self._filter: Optional[Set[str]] = None
        if class_filter:
            self._filter = {c.strip().lower() for c in class_filter if c.strip()}

    def detect(self, image_bgr: np.ndarray) -> List[BlockDetection]:
        results = self._model.predict(
            image_bgr, conf=self._conf, verbose=False, device=self._device
        )
        out: List[BlockDetection] = []
        if not results:
            return out
        r0 = results[0]
        names = getattr(r0, "names", {}) or {}
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return out
        for b in boxes:
            xyxy = b.xyxy[0].tolist()
            x0, y0, x1, y1 = (int(v) for v in xyxy)
            w, h = max(1, x1 - x0), max(1, y1 - y0)
            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            label = str(names.get(cls_id, f"class_{cls_id}"))
            if self._filter is not None and label.lower() not in self._filter:
                continue
            score = float(b.conf[0].item()) if b.conf is not None else 0.0
            out.append(
                BlockDetection(
                    label=label,
                    confidence=score,
                    cx=(x0 + x1) * 0.5,
                    cy=(y0 + y1) * 0.5,
                    x=x0,
                    y=y0,
                    w=w,
                    h=h,
                )
            )
        out.sort(key=lambda d: d.confidence * (d.w * d.h), reverse=True)
        return out


def parse_class_filter(text: str) -> Optional[List[str]]:
    """'bottle,cup' → list; empty → None (no filter)."""
    text = (text or "").strip()
    if not text or text.lower() in ("*", "all"):
        return None
    return [p.strip() for p in text.split(",") if p.strip()]


def default_models_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def resolve_yolo_weights(explicit: str = "") -> Optional[Path]:
    """Prefer explicit path, else package models/yolov8n.pt."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        return None
    cand = default_models_dir() / "yolov8n.pt"
    if cand.is_file():
        return cand
    return None
