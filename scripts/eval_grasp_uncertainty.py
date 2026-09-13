#!/usr/bin/env python3
"""Ablation: heuristic grasp rank vs risk-adjusted uncertainty rank.

Metric that matters with same-label competitors:
  picked_primary_rate — chose workspace primary vs edge distractor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from arm_system.block_detector import detect_blocks
from arm_system.grasp_ranking import rank_grasp_candidates
from arm_system.grasp_uncertainty import rank_grasp_candidates_uncertain
from arm_system.hard_scene import make_hard_scene
from arm_system.metrics import MetricsLogger, TrialRecord
from arm_system.pose_from_bbox import estimate_camera_pose, pixel_error


def results_dir() -> Path:
    return _PKG / "results"


def _pick_primary_and_competitor(scene):
    reds = [o for o in scene.objects if o.label == "red"]
    primary = reds[0]
    competitor = reds[1] if len(reds) > 1 else None
    return primary, competitor


def _is_closer_to_primary(det, primary, competitor) -> bool:
    if competitor is None:
        return True
    dp = (det.cx - primary.cx) ** 2 + (det.cy - primary.cy) ** 2
    dc = (det.cx - competitor.cx) ** 2 + (det.cy - competitor.cy) ** 2
    return dp <= dc


def _eval_mode(mode: str, trials: int, occ: float) -> dict:
    logger = MetricsLogger(results_dir(), run_name=f"grasp_{mode}")
    primary_hits = 0
    for i in range(trials):
        scene = make_hard_scene(
            seed=2000 + i,
            t=i * 0.41,
            occlusion_prob=occ,
            fx=600.0,
            object_size_m=0.06,
            competing_same_label=True,
        )
        primary, competitor = _pick_primary_and_competitor(scene)
        dets = detect_blocks(scene.image_bgr, min_area=200)
        if mode == "heuristic":
            ranked = rank_grasp_candidates(
                dets, image_w=640, image_h=480, target_label="red", top_k=5
            )
            cand = next((s.det for s in ranked if s.det.label == "red"), None)
            score = next((s.score for s in ranked if s.det.label == "red"), 0.0)
            unc = None
        else:
            ranked_u = rank_grasp_candidates_uncertain(
                dets,
                image_w=640,
                image_h=480,
                target_label="red",
                top_k=5,
                fx=600.0,
                object_size_m=0.06,
                lambda_risk=1.6,
            )
            cand = next(
                (g.scored.det for g in ranked_u if g.scored.det.label == "red"), None
            )
            score = next(
                (
                    g.risk_adjusted_score
                    for g in ranked_u
                    if g.scored.det.label == "red"
                ),
                0.0,
            )
            unc = next(
                (g.uncertainty for g in ranked_u if g.scored.det.label == "red"),
                None,
            )

        hit = bool(cand is not None and _is_closer_to_primary(cand, primary, competitor))
        if hit:
            primary_hits += 1

        rec = TrialRecord(
            trial_id=i,
            timestamp=time.time(),
            scene_mode=mode,
            target_label="red",
            detected=cand is not None,
            occluded_gt=primary.occluded,
            extra={
                "score": score,
                "uncertainty": unc,
                "picked_primary": hit,
            },
        )
        if cand is None:
            rec.failure_mode = "no_detection"
            logger.log(rec)
            continue
        # Evaluate pose vs the GT we actually aimed at (primary if hit else nearest)
        gt = primary if hit else (competitor or primary)
        est = estimate_camera_pose(
            cand, fx=600, fy=600, cx0=320, cy0=240, object_size_m=0.06
        )
        pe = pixel_error(est.u, est.v, gt.cx, gt.cy)
        de = abs(est.z_m - gt.z_m)
        rec.pixel_error = pe
        rec.depth_error_m = de
        rec.pose_method = est.method
        if not hit:
            rec.failure_mode = "wrong_instance"
        elif pe > 25.0 or de > 0.08:
            rec.failure_mode = "pose_inaccurate"
        logger.log(rec)

    s = logger.summarize()
    s["picked_primary_rate"] = primary_hits / max(trials, 1)
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--occlusion", type=float, default=0.75)
    args = p.parse_args(argv)

    h = _eval_mode("heuristic", args.trials, args.occlusion)
    u = _eval_mode("uncertain", args.trials, args.occlusion)
    out = {
        "occlusion": args.occlusion,
        "heuristic": h,
        "uncertain": u,
        "delta_picked_primary": u["picked_primary_rate"] - h["picked_primary_rate"],
        "note": "Positive delta_picked_primary = uncertain better at avoiding edge distractors",
    }
    path = results_dir() / "grasp_uncertainty_ablation.json"
    results_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
