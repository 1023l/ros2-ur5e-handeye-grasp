"""Episode / step schema for AGV roll-transfer demos and eval logs."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from arm_system.policy.constants import SCHEMA_VERSION, TASK_ID


@dataclass
class StepRecord:
    t: int
    image: str  # relative path under episode dir
    state: List[float]
    action: List[float]
    ee_xyz: List[float]
    gripper: float
    phase: str  # approach|grasp|lift|transit|place|retreat|home
    geom_prior: Optional[List[float]] = None  # geometric residual teacher action


@dataclass
class EpisodeMeta:
    episode_id: str
    task_id: str = TASK_ID
    schema_version: int = SCHEMA_VERSION
    success: bool = False
    failure_mode: str = ""
    profile: str = "nominal"
    seed: int = 0
    n_steps: int = 0
    roll_xyz: List[float] = field(default_factory=list)
    agv_bay_xy: List[float] = field(default_factory=list)
    perturb: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    # expert true bay − map/belief bay (what residual ACT should recover from image)
    dock_offset_xy: List[float] = field(default_factory=lambda: [0.0, 0.0])
    created_at: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    meta: EpisodeMeta
    steps: List[StepRecord] = field(default_factory=list)
    root: Optional[Path] = None

    def save(self, root: Path) -> Path:
        root = Path(root)
        ep_dir = root / self.meta.episode_id
        img_dir = ep_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for s in self.steps:
            # images already written by collector; ensure path is relative
            if not s.image.startswith("images/"):
                s.image = f"images/{Path(s.image).name}"
        payload = {
            "meta": asdict(self.meta),
            "steps": [asdict(s) for s in self.steps],
        }
        (ep_dir / "episode.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.root = ep_dir
        return ep_dir

    @staticmethod
    def load(ep_dir: Path) -> "Episode":
        ep_dir = Path(ep_dir)
        data = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
        meta = EpisodeMeta(**data["meta"])
        steps = [StepRecord(**s) for s in data["steps"]]
        return Episode(meta=meta, steps=steps, root=ep_dir)


def new_episode_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"ep_{stamp}_{uuid.uuid4().hex[:8]}"


def save_image(ep_dir: Path, t: int, image_bgr: np.ndarray) -> str:
    img_dir = ep_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rel = f"images/{t:04d}.png"
    import cv2

    cv2.imwrite(str(ep_dir / rel), image_bgr)
    return rel


class DatasetIndex:
    """Manifest over a demo root for train/eval splits."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "manifest.jsonl"

    def append(self, meta: EpisodeMeta) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")

    def list_episodes(self, *, success_only: bool = True) -> List[Path]:
        eps: List[Path] = []
        if not self.path.exists():
            # fallback: scan dirs
            for p in sorted(self.root.glob("ep_*/episode.json")):
                ep = Episode.load(p.parent)
                if success_only and not ep.meta.success:
                    continue
                eps.append(p.parent)
            return eps
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if success_only and not m.get("success", False):
                continue
            eps.append(self.root / m["episode_id"])
        return eps
