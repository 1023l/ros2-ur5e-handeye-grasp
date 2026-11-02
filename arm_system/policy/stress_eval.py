"""Dock noise stress curves: geometric vs ACT controller modes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from arm_system.policy.eval_protocol import run_act_episode, run_geom_episode_as_trial
from arm_system.policy.scene_agv import ScenePerturb
from arm_system.policy.train_loop import load_policy


def sweep_dock_stress(
    ckpt_dir: Path,
    out_dir: Path,
    *,
    noise_levels: Optional[List[float]] = None,
    trials: int = 30,
    seed0: int = 5000,
    controller_modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    noise_levels = noise_levels or [0.0, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075]
    controller_modes = controller_modes or [
        "geometric",
        "nominal",
        "vision_only",
        "dock_only",
        "fusion",
    ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, stats, residual, device = load_policy(ckpt_dir)
    rows: List[Dict[str, Any]] = []

    for noise in noise_levels:
        perturb = ScenePerturb(
            roll_xy_noise=0.01,
            agv_dock_xy_noise=float(noise),
            perception_xy_noise=0.012,
            occlusion=0.12,
            clutter=2,
        )
        row: Dict[str, Any] = {"agv_dock_xy_noise": noise}
        base_seed = seed0 + int(noise * 10000)
        for mode in controller_modes:
            ok = 0
            for i in range(trials):
                seed = base_seed + i
                if mode == "geometric":
                    tr = run_geom_episode_as_trial(
                        seed, profile="nominal", perturb_override=perturb
                    )
                else:
                    tr = run_act_episode(
                        model=model,
                        stats=stats,
                        residual=residual,
                        device=device,
                        seed=seed,
                        profile="nominal",
                        perturb_override=perturb,
                        dock_mode=mode,
                    )
                if tr.success:
                    ok += 1
            row[mode] = ok / max(1, trials)
        rows.append(row)

    summary = {
        "trials_per_point": trials,
        "noise_levels": noise_levels,
        "modes": controller_modes,
        "rows": rows,
        "ckpt": str(ckpt_dir),
        "created_at": time.time(),
    }
    (out_dir / "stress_curve.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_stress_md(out_dir / "STRESS_CURVE.md", summary)
    return summary


def _write_stress_md(path: Path, summary: Dict[str, Any]) -> None:
    modes = summary["modes"]
    lines = [
        "# AGV dock noise stress curve",
        "",
        f"- trials/point: {summary['trials_per_point']}",
        f"- ckpt: `{summary.get('ckpt')}`",
        "",
        "| dock_noise (m) | " + " | ".join(modes) + " |",
        "|---|" + "|".join(["---"] * len(modes)) + "|",
    ]
    for row in summary["rows"]:
        n = row["agv_dock_xy_noise"]
        cells = " | ".join(f"{row[m] * 100:.1f}%" for m in modes)
        lines.append(f"| {n:.3f} | {cells} |")
    lines.append("")
    lines.append(
        "Read the cliff: where geometric drops while fusion stays high — that is your interview story."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
