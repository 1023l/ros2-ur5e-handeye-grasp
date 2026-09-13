#!/usr/bin/env python3
"""Smoke: one YOLO inference on synthetic photo scene → print boxes."""

from __future__ import annotations

import sys

import cv2

from arm_system.block_detector import annotate, make_synthetic_photo_scene
from arm_system.yolo_detector import YoloDetector, default_models_dir, resolve_yolo_weights


def main() -> int:
    weights = resolve_yolo_weights("")
    if weights is None:
        print("No yolov8n.pt — run: ros2 run arm_system setup_yolo_weights.py")
        return 2
    patch_path = default_models_dir() / "demo_target.jpg"
    if not patch_path.is_file():
        print(f"Missing {patch_path} — run setup_yolo_weights.py")
        return 2
    patch = cv2.imread(str(patch_path))
    img = make_synthetic_photo_scene(640, 480, patch, cx=320, cy=240, patch_h=160)
    det = YoloDetector(str(weights), conf=0.25, class_filter=None, device="cpu")
    boxes = det.detect(img)
    print(f"weights={weights}")
    print(f"detections={len(boxes)}")
    for b in boxes[:8]:
        print(f"  {b.label} conf={b.confidence:.2f} cx={b.cx:.0f} cy={b.cy:.0f}")
    if not boxes:
        print("FAIL: expected at least one detection")
        return 1
    out = default_models_dir() / "yolo_smoke_debug.jpg"
    cv2.imwrite(str(out), annotate(img, boxes))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
