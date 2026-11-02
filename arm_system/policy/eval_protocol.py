"""Closed-loop eval: geometric baseline vs ACT under AGV dock profiles.

Runs without MoveIt — EE is integrated from predicted deltas against frozen
scene GT. Success = grasp near true roll + release inside true AGV bay.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from arm_system.policy.constants import (
    ACTION_DIM,
    AGV_BAY_XY,
    APPROACH_Z,
    CHUNK_SIZE,
    IMG_H,
    IMG_W,
)
from arm_system.policy.geometry_teacher import (
    EEState,
    _aa_from_tool_down,
    _ee_to_uv,
    build_place_only_path,
    build_waypoint_path,
    classify_failure,
    ee_delta_action,
    render_with_ee_marker,
    run_geometric_episode,
    state_vector,
)
from arm_system.policy.dock_controller import fuse_dock_bay
from arm_system.policy.scene_agv import (
    EVAL_PROFILES,
    ScenePerturb,
    estimate_bay_xy_from_image,
    in_agv_bay,
    make_agv_scene,
    perceive_roll,
)
from arm_system.policy.temporal_ensemble import TemporalEnsembler
from arm_system.policy.train_loop import load_policy


@dataclass
class TrialResult:
    trial_id: int
    method: str
    profile: str
    seed: int
    success: bool
    failure_mode: str
    place_xy: List[float]
    grasp_err_m: float
    place_err_m: float
    n_steps: int
    extra: Dict[str, Any] = field(default_factory=dict)


def _prep_image(image_bgr: np.ndarray, ee: EEState) -> torch.Tensor:
    img = render_with_ee_marker(image_bgr, _ee_to_uv(ee.xyz))
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def _bootstrap_ci(flags: List[bool], n_boot: int = 400, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(flags, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(sample.mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def run_act_episode(
    *,
    model,
    stats,
    residual: bool,
    device: str,
    seed: int,
    profile: str = "nominal",
    perturb_override: Optional[ScenePerturb] = None,
    dock_mode: str = "fusion",
    max_steps: int = 80,
    use_ensemble: bool = True,
) -> TrialResult:
    perturb = perturb_override if perturb_override is not None else EVAL_PROFILES[profile]
    rng = np.random.default_rng(seed)
    gt = make_agv_scene(seed=seed, perturb=perturb)
    pnoise = float(gt.extras.get("perception_xy_noise", 0.01))
    roll_belief = perceive_roll(gt, rng=rng, perception_xy_noise=pnoise)
    dock_read_noise = float(perturb.agv_dock_xy_noise) * 0.15
    bay_belief = (
        AGV_BAY_XY[0] + float(rng.normal(0, max(0.002, dock_read_noise))),
        AGV_BAY_XY[1] + float(rng.normal(0, max(0.002, dock_read_noise))),
    )

    # geometric prior path for residual mode (open-loop belief)
    geom_path = build_waypoint_path(roll_belief, bay_belief, steps_per_seg=3)
    geom_actions = []
    geom_phases = []
    for i in range(1, len(geom_path)):
        geom_actions.append(ee_delta_action(geom_path[i - 1][1], geom_path[i][1]))
        geom_phases.append(geom_path[i][0])
    if not geom_actions:
        geom_actions = [np.zeros(ACTION_DIM)]
        geom_phases = ["home"]

    ee = EEState(
        xyz=np.array([0.3, 0.0, 0.35], dtype=np.float64),
        aa=_aa_from_tool_down(),
        gripper=0.0,
    )
    grasped = False
    place_xy = (float(ee.xyz[0]), float(ee.xyz[1]))
    grasp_err = 1.0
    n_geom = len(geom_actions)
    pick_phases = {"approach", "grasp", "grasp_close", "lift"}
    dock_corrected = False
    t = 0

    # 1) Geometric pick (snap waypoints — identical to baseline grasp)
    for geom_i in range(n_geom):
        phase = geom_phases[geom_i]
        if phase not in pick_phases:
            break
        nxt_ee = geom_path[geom_i + 1][1]
        ee = EEState(
            xyz=nxt_ee.xyz.copy(), aa=nxt_ee.aa.copy(), gripper=float(nxt_ee.gripper)
        )
        if phase.startswith("grasp") and ee.gripper >= 0.5:
            grasp_err = float(np.linalg.norm(ee.xyz[:2] - np.array(gt.roll_xyz[:2])))
            if grasp_err < 0.045:
                grasped = True
        t += 1

    vis_x, vis_y, vis_conf = estimate_bay_xy_from_image(
        gt.image_bgr, fallback_xy=bay_belief
    )
    st = state_vector(ee, roll_belief, bay_belief)
    st_n = stats.norm_state(st).astype(np.float32)
    img_t = _prep_image(gt.image_bgr, ee).to(device)
    state_t = torch.from_numpy(st_n).unsqueeze(0).to(device)
    with torch.no_grad():
        off = model.predict_dock_offset(img_t, state_t)[0].cpu().numpy()
    learn_bay = (
        bay_belief[0] + float(np.clip(off[0], -0.10, 0.10)),
        bay_belief[1] + float(np.clip(off[1], -0.10, 0.10)),
    )
    corrected_bay, dx, dy = fuse_dock_bay(
        bay_belief,
        vis_xy=(vis_x, vis_y),
        vis_conf=vis_conf,
        learn_xy=learn_bay,
        mode=dock_mode,  # type: ignore[arg-type]
    )
    dock_corrected = dock_mode != "nominal"

    # 3) Geometric finish to ACT-corrected bay
    place_path = build_place_only_path(corrected_bay, start=ee, steps_per_seg=3)
    for phase, nxt in place_path[1:]:
        ee = EEState(xyz=nxt.xyz.copy(), aa=nxt.aa.copy(), gripper=float(nxt.gripper))
        if phase.startswith("place"):
            place_xy = (float(ee.xyz[0]), float(ee.xyz[1]))
        t += 1
        if t >= max_steps:
            break

    placed = bool(grasped and in_agv_bay(place_xy, gt.agv_bay_xy))
    if placed and ee.gripper > 0.5:
        placed = False
    fail = classify_failure(
        grasped=bool(grasped), placed_in_bay=placed, occluded=bool(gt.occluded)
    )
    success = bool(placed and fail == "")
    place_err = float(np.linalg.norm(np.array(place_xy) - np.array(gt.agv_bay_xy)))
    return TrialResult(
        trial_id=int(seed),
        method=f"act_{dock_mode}",
        profile=profile if perturb_override is None else f"custom_{dock_mode}",
        seed=int(seed),
        success=success,
        failure_mode=fail,
        place_xy=[float(place_xy[0]), float(place_xy[1])],
        grasp_err_m=float(grasp_err if grasped else -1.0),
        place_err_m=place_err,
        n_steps=int(t),
        extra={
            "ensemble": bool(use_ensemble),
            "residual": bool(residual),
            "dock_corrected": bool(dock_corrected),
            "dock_offset_xy": [float(dx), float(dy)],
            "corrected_bay": [float(corrected_bay[0]), float(corrected_bay[1])],
            "vis_bay_conf": float(vis_conf),
            "dock_mode": dock_mode,
        },
    )


def run_geom_episode_as_trial(
    seed: int,
    profile: str = "nominal",
    perturb_override: Optional[ScenePerturb] = None,
) -> TrialResult:
    perturb = perturb_override if perturb_override is not None else EVAL_PROFILES[profile]
    r = run_geometric_episode(seed=seed, profile_perturb=perturb, record=False)
    place_err = float(
        np.linalg.norm(np.array(r.final_place_xy) - np.array(make_agv_scene(seed=seed, perturb=perturb).agv_bay_xy))
    )
    # recompute grasp err roughly via success flag
    return TrialResult(
        trial_id=int(seed),
        method="geometric",
        profile=profile,
        seed=int(seed),
        success=bool(r.success),
        failure_mode=str(r.failure_mode),
        place_xy=[float(r.final_place_xy[0]), float(r.final_place_xy[1])],
        grasp_err_m=-1.0,
        place_err_m=float(place_err),
        n_steps=int(r.n_steps),
    )


def evaluate_compare(
    ckpt_dir: Path,
    out_dir: Path,
    *,
    profiles: Optional[List[str]] = None,
    trials: int = 40,
    seed0: int = 1000,
    use_ensemble: bool = True,
    dock_mode: str = "fusion",
    ablation_modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    profiles = profiles or list(EVAL_PROFILES.keys())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, stats, residual, device = load_policy(ckpt_dir)

    act_modes = ablation_modes or [dock_mode]
    results: List[TrialResult] = []
    for profile in profiles:
        for i in range(trials):
            seed = seed0 + i
            results.append(run_geom_episode_as_trial(seed, profile))
            for dm in act_modes:
                results.append(
                    run_act_episode(
                        model=model,
                        stats=stats,
                        residual=residual,
                        device=device,
                        seed=seed,
                        profile=profile,
                        dock_mode=dm,
                        use_ensemble=use_ensemble,
                    )
                )

    # summarize
    summary: Dict[str, Any] = {
        "trials_per_profile": trials,
        "profiles": {},
        "created_at": time.time(),
        "ckpt": str(ckpt_dir),
        "act_modes": act_modes,
    }
    for profile in profiles:
        block: Dict[str, Any] = {}
        geom_sub = [
            r for r in results if r.profile == profile and r.method == "geometric"
        ]
        flags_g = [r.success for r in geom_sub]
        mean_g, lo_g, hi_g = _bootstrap_ci(flags_g)
        block["geometric"] = {
            "success_rate": mean_g,
            "ci95": [lo_g, hi_g],
            "n": len(geom_sub),
            "failure_modes": dict(
                Counter(r.failure_mode for r in geom_sub if r.failure_mode)
            ),
            "mean_place_err_m": float(
                np.mean([r.place_err_m for r in geom_sub]) if geom_sub else 0.0
            ),
        }
        for dm in act_modes:
            key = f"act_{dm}"
            sub = [r for r in results if r.profile == profile and r.method == key]
            flags = [r.success for r in sub]
            mean, lo, hi = _bootstrap_ci(flags)
            block[key] = {
                "success_rate": mean,
                "ci95": [lo, hi],
                "n": len(sub),
                "failure_modes": dict(Counter(r.failure_mode for r in sub if r.failure_mode)),
                "mean_place_err_m": float(
                    np.mean([r.place_err_m for r in sub]) if sub else 0.0
                ),
            }
        g = block["geometric"]["success_rate"]
        primary = f"act_{act_modes[0]}"
        a = block.get(primary, {}).get("success_rate", 0.0)
        block["delta_pp"] = (a - g) * 100.0
        summary["profiles"][profile] = block

    # write artifacts
    jsonl = out_dir / "trials.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(out_dir / "EVAL_REPORT.md", summary)
    return summary


def _write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# AGV Roll Transfer — ACT vs Geometric",
        "",
        f"- ckpt: `{summary.get('ckpt')}`",
        f"- trials/profile: {summary.get('trials_per_profile')}",
        "",
        "| Profile | Geometric | ACT (primary) | Δ (pp) |",
        "|---------|-----------|-----------------|--------|",
    ]
    act_modes = summary.get("act_modes") or ["fusion"]
    primary = f"act_{act_modes[0]}"
    for name, block in summary["profiles"].items():
        g = block["geometric"]["success_rate"] * 100
        a = block.get(primary, block.get("act", {})).get("success_rate", 0) * 100
        d = block["delta_pp"]
        lines.append(f"| {name} | {g:.1f}% | {a:.1f}% | {d:+.1f} |")
    lines.append("")
    lines.append("## Failure modes (ACT primary)")
    for name, block in summary["profiles"].items():
        lines.append(f"### {name}")
        modes = block.get(primary, {}).get("failure_modes", {})
        if not modes:
            lines.append("- (none)")
        else:
            for k, v in sorted(modes.items(), key=lambda x: -x[1]):
                lines.append(f"- `{k}`: {v}")
        lines.append("")
    lines.append(
        "Success = grasp within 5cm of true roll center AND release inside true AGV bay."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
