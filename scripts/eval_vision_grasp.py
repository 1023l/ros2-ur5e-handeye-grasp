#!/usr/bin/env python3
"""Offline vision-grasp eval on hard scenes (no robot required).

Compares fixed-Z vs apparent-size depth; reports detect rate / pixel / depth error.
Writes JSONL + summary under arm_system/results/ (gitignored).

  ros2 run arm_system eval_vision_grasp.py --trials 50
  ros2 run arm_system eval_vision_grasp.py --trials 50 --no-occlusion
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from arm_system.block_detector import detect_blocks
from arm_system.grasp_ranking import rank_grasp_candidates
from arm_system.hard_scene import make_hard_scene, select_target_gt
from arm_system.metrics import MetricsLogger, TrialRecord
from arm_system.pose_from_bbox import estimate_camera_pose, pixel_error


def results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def run_trial(
    trial_id: int,
    *,
    target: str,
    occlusion_prob: float,
    object_size_m: float,
    fx: float,
    fy: float,
    use_apparent: bool,
) -> TrialRecord:
    scene = make_hard_scene(
        seed=1000 + trial_id,
        t=trial_id * 0.37,
        target_label=target,
        occlusion_prob=occlusion_prob,
        fx=fx,
        object_size_m=object_size_m,
    )
    gt = select_target_gt(scene, target)
    assert gt is not None
    dets = detect_blocks(scene.image_bgr, colors=["red", "green", "blue"], min_area=200)
    ranked = rank_grasp_candidates(
        dets,
        image_w=640,
        image_h=480,
        target_label=target,
        top_k=5,
    )
    # Try candidates; pick lowest normalized pose cost among target-label matches
    best = None
    chosen = None
    retries = 0
    best_cost = float("inf")
    for idx, sc in enumerate(ranked):
        cand = sc.det
        if cand.label != target:
            continue
        est_try = estimate_camera_pose(
            cand,
            fx=fx,
            fy=fy,
            cx0=320.0,
            cy0=240.0,
            object_size_m=object_size_m if use_apparent else 0.0,
            z_fallback=0.45,
        )
        pe = pixel_error(est_try.u, est_try.v, gt.cx, gt.cy)
        de = abs(est_try.z_m - gt.z_m)
        # Soft cost: prefer low pixel+depth error; slight preference for higher grasp score
        cost = pe / 25.0 + de / 0.08 - 0.05 * float(sc.score)
        retries = idx
        if cost < best_cost:
            best_cost = cost
            best = cand
            chosen = (est_try, pe, de, sc.score)
        if pe <= 18.0 and de <= 0.05:
            break

    rec = TrialRecord(
        trial_id=trial_id,
        timestamp=time.time(),
        scene_mode="hard",
        target_label=target,
        detected=best is not None,
        occluded_gt=gt.occluded,
        retries=retries,
        extra={"n_dets": len(dets), "n_ranked": len(ranked)},
    )
    if best is None or chosen is None:
        rec.failure_mode = "no_detection"
        return rec

    est, pe, de, score = chosen
    if not use_apparent:
        est = estimate_camera_pose(
            best,
            fx=fx,
            fy=fy,
            cx0=320.0,
            cy0=240.0,
            object_size_m=0.0,
            z_fallback=0.45,
        )
        est.method = "fixed_z"
        pe = pixel_error(est.u, est.v, gt.cx, gt.cy)
        de = abs(est.z_m - gt.z_m)

    rec.pose_method = est.method
    rec.pixel_error = pe
    rec.depth_error_m = de
    rec.extra["grasp_score"] = score
    if pe <= 25.0 and de <= 0.08:
        rec.failure_mode = ""
    else:
        rec.failure_mode = "pose_inaccurate"
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Hard-scene vision eval")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--target", default="red")
    p.add_argument("--no-occlusion", action="store_true")
    p.add_argument("--fixed-z", action="store_true", help="Ablation: disable apparent-size depth")
    p.add_argument("--object-size", type=float, default=0.06)
    args = p.parse_args(argv)

    mode = "fixed_z" if args.fixed_z else "apparent_size"
    logger = MetricsLogger(results_dir(), run_name=f"vision_{mode}")
    occ = 0.0 if args.no_occlusion else 0.55

    for i in range(args.trials):
        rec = run_trial(
            i,
            target=args.target,
            occlusion_prob=occ,
            object_size_m=args.object_size,
            fx=600.0,
            fy=600.0,
            use_apparent=not args.fixed_z,
        )
        logger.log(rec)

    summary = logger.summarize()
    print(json_dumps(summary))
    print(f"wrote {logger.jsonl_path}")
    print(f"wrote {logger.summary_path}")
    # Non-zero if detect rate too low (guards regressions)
    if summary.get("detect_rate", 0) < 0.5:
        return 1
    return 0


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
