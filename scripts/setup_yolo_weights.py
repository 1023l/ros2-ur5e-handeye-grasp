#!/usr/bin/env python3
"""Download YOLOv8n weights + a demo photo crop for synthetic YOLO scenes."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


def models_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  → {dest} ({dest.stat().st_size} bytes)")


def ensure_weights(dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"Weights OK: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Official ultralytics release asset
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    try:
        download(url, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"Direct download failed ({exc}), trying ultralytics API...")
        from ultralytics import YOLO  # type: ignore

        m = YOLO("yolov8n.pt")
        # After load, weight usually lands in CWD
        for c in (Path("yolov8n.pt"), Path(str(getattr(m, "ckpt_path", "")))):
            if c.is_file():
                dest.write_bytes(c.read_bytes())
                break
        else:
            raise RuntimeError("Could not locate yolov8n.pt after YOLO() load") from exc
    print(f"Weights saved: {dest}")
    return dest


def ensure_demo_patch(dest: Path, weights: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 1000:
        print(f"Demo patch OK: {dest}")
        return dest
    import cv2
    from ultralytics import YOLO  # type: ignore

    sample = dest.parent / "_bus.jpg"
    if not sample.is_file():
        urls = [
            "https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg",
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
            "https://ultralytics.com/images/bus.jpg",
        ]
        last_err: Exception | None = None
        for url in urls:
            try:
                download(url, sample)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"  retry ({exc})")
        if last_err is not None or not sample.is_file():
            # Last resort: let YOLO fetch via its own downloader on a URL predict
            print("Trying ultralytics predict-on-URL to cache sample...")
            model = YOLO(str(weights))
            results = model.predict(
                "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
                conf=0.25,
                verbose=False,
            )
            # results[0].orig_img is the loaded image
            img = results[0].orig_img
            if img is None:
                raise RuntimeError(f"Could not fetch demo image: {last_err}")
            cv2.imwrite(str(sample), img)
    img = cv2.imread(str(sample))
    if img is None:
        raise RuntimeError(f"Failed to read {sample}")
    model = YOLO(str(weights))
    results = model.predict(img, conf=0.25, verbose=False)
    best = None
    names = results[0].names if results else {}
    prefer = {"person", "bus", "car", "truck", "bottle", "cup", "sports ball"}
    for b in results[0].boxes:
        cls_id = int(b.cls[0].item())
        label = str(names.get(cls_id, ""))
        conf = float(b.conf[0].item())
        x0, y0, x1, y1 = (int(v) for v in b.xyxy[0].tolist())
        area = max(1, (x1 - x0) * (y1 - y0))
        score = conf * (2.0 if label.lower() in prefer else 1.0) * area
        if best is None or score > best[0]:
            best = (score, label, x0, y0, x1, y1)
    h, w = img.shape[:2]
    if best is None:
        x0, y0, x1, y1 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
        label = "unknown"
    else:
        _, label, x0, y0, x1, y1 = best
        pad = 8
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    crop = img[y0:y1, x0:x1]
    cv2.imwrite(str(dest), crop)
    dest.with_suffix(".txt").write_text(f"{label}\n", encoding="utf-8")
    print(f"Demo patch: {dest} label={label} shape={crop.shape}")
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path, default=None)
    args = p.parse_args(argv)
    root = args.models_dir or models_dir()
    root.mkdir(parents=True, exist_ok=True)
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ultralytics missing. Install CPU stack:")
        print(
            '  pip3 install --user torch torchvision '
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        print('  pip3 install --user "ultralytics>=8.0.0,<8.4"')
        return 2
    weights = ensure_weights(root / "yolov8n.pt")
    ensure_demo_patch(root / "demo_target.jpg", weights)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
