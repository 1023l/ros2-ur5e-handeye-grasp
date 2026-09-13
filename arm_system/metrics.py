"""Trial metrics logging for vision-grasp evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrialRecord:
    trial_id: int
    timestamp: float
    scene_mode: str
    target_label: str
    detected: bool
    occluded_gt: bool = False
    pixel_error: Optional[float] = None
    depth_error_m: Optional[float] = None
    pose_method: str = ""
    retries: int = 0
    plan_success: Optional[bool] = None
    exec_success: Optional[bool] = None
    failure_mode: str = ""
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class MetricsLogger:
    def __init__(self, out_dir: Path, run_name: str = "") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = run_name or f"run_{stamp}"
        self.jsonl_path = self.out_dir / f"{name}.jsonl"
        self.summary_path = self.out_dir / f"{name}_summary.json"
        self.records: List[TrialRecord] = []

    def log(self, rec: TrialRecord) -> None:
        self.records.append(rec)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def summarize(self) -> Dict[str, Any]:
        n = len(self.records)
        if n == 0:
            summary: Dict[str, Any] = {"n": 0}
        else:
            det = sum(1 for r in self.records if r.detected)
            pix = [r.pixel_error for r in self.records if r.pixel_error is not None]
            dep = [r.depth_error_m for r in self.records if r.depth_error_m is not None]
            plan_ok = [r for r in self.records if r.plan_success is True]
            exec_ok = [r for r in self.records if r.exec_success is True]
            modes: Dict[str, int] = {}
            for r in self.records:
                if r.failure_mode:
                    modes[r.failure_mode] = modes.get(r.failure_mode, 0) + 1
            summary = {
                "n": n,
                "detect_rate": det / n,
                "mean_pixel_error": sum(pix) / len(pix) if pix else None,
                "mean_depth_error_m": sum(dep) / len(dep) if dep else None,
                "plan_success_rate": len(plan_ok) / n if any(r.plan_success is not None for r in self.records) else None,
                "exec_success_rate": len(exec_ok) / n if any(r.exec_success is not None for r in self.records) else None,
                "failure_modes": modes,
                "jsonl": str(self.jsonl_path),
            }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary
