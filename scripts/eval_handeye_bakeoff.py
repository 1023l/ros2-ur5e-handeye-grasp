#!/usr/bin/env python3
"""Hand-eye method bake-off with mean/std — senior-style baseline comparison.

Compares OpenCV Tsai / Park / Daniilidis vs +SE3-BA vs +Ceres-consistency-gate
under the same noise. Output is a decision table, not a single magic number.

  python3 scripts/eval_handeye_bakeoff.py --trials 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from arm_system.handeye_ba import refine_eye_to_hand_ba
from arm_system.handeye_ceres_style import (
    refine_handeye_consistency_lm,
    synthesize_consistency_samples_eye_to_hand,
)
from arm_system.handeye_math import make_T, rpy_to_R, solve_eye_to_hand, synthesize_eye_to_hand_samples


def results_dir() -> Path:
    return _PKG / "results"


def _stats(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
    }


def _opencv_methods() -> List[Tuple[str, int]]:
    import cv2

    out = [("park", cv2.CALIB_HAND_EYE_PARK)]
    if hasattr(cv2, "CALIB_HAND_EYE_TSAI"):
        out.append(("tsai", cv2.CALIB_HAND_EYE_TSAI))
    if hasattr(cv2, "CALIB_HAND_EYE_DANIILIDIS"):
        out.append(("daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS))
    if hasattr(cv2, "CALIB_HAND_EYE_HORAUD"):
        out.append(("horaud", cv2.CALIB_HAND_EYE_HORAUD))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--noise-t", type=float, default=0.004)
    p.add_argument("--noise-r", type=float, default=0.02)
    args = p.parse_args(argv)

    R_gt = rpy_to_R(math.pi, 0.0, 0.0)
    t_gt = np.array([0.40, 0.0, 0.50])
    T_gt = make_T(R_gt, t_gt)
    T_ee_board = make_T(np.eye(3), [0.0, 0.0, 0.08])

    buckets: Dict[str, List[float]] = {}
    for name, _ in _opencv_methods():
        buckets[name] = []
    buckets["park+ba"] = []
    buckets["park+ceres_gate"] = []

    for i in range(args.trials):
        rng = np.random.default_rng(400 + i)
        T_ees = []
        for _ in range(args.samples):
            R = rpy_to_R(
                float(rng.uniform(-0.3, 0.3)),
                float(rng.uniform(-0.4, 0.2)),
                float(rng.uniform(-0.8, 0.8)),
            )
            t = np.array(
                [
                    float(rng.uniform(0.25, 0.55)),
                    float(rng.uniform(-0.25, 0.25)),
                    float(rng.uniform(0.15, 0.45)),
                ]
            )
            T_ees.append(make_T(R, t))
        T_be, T_cb = synthesize_eye_to_hand_samples(
            T_gt,
            T_ee_board,
            T_ees,
            noise_t=args.noise_t,
            noise_r_rad=args.noise_r,
            rng=rng,
        )
        for name, method in _opencv_methods():
            try:
                R, t = solve_eye_to_hand(T_be, T_cb, method=method)
                buckets[name].append(float(np.linalg.norm(t - t_gt)))
            except Exception:
                buckets[name].append(float("nan"))

        R_p, t_p = solve_eye_to_hand(T_be, T_cb)  # default Park
        T_p = make_T(R_p, t_p)
        ba = refine_eye_to_hand_ba(T_p, T_be, T_cb, T_ee_board)
        buckets["park+ba"].append(float(np.linalg.norm(ba.T_base_cam[:3, 3] - t_gt)))

        samples = synthesize_consistency_samples_eye_to_hand(
            T_gt,
            T_ee_board,
            T_ees,
            noise_t=args.noise_t,
            noise_r=args.noise_r,
            rng=rng,
        )
        T_c, _, _ = refine_handeye_consistency_lm(T_p, samples, eye_to_hand=True)
        buckets["park+ceres_gate"].append(float(np.linalg.norm(T_c[:3, 3] - t_gt)))

    table: Dict[str, Any] = {}
    for k, xs in buckets.items():
        clean = [x for x in xs if x == x]  # drop nan
        if not clean:
            continue
        table[k] = _stats(clean)
        table[k]["n"] = len(clean)

    # Decision: lowest mean wins; BA expected champion under this residual family
    ranked = sorted(table.items(), key=lambda kv: kv[1]["mean"])
    recommendation = {
        "best_by_mean_err_t": ranked[0][0],
        "ranking": [k for k, _ in ranked],
        "rationale": (
            "Under synthetic eye-to-hand noise, SE3 pose-chain BA usually dominates "
            "closed-form OpenCV. Ceres-style consistency+gate is for industrial residual "
            "alignment / safety (don't ship a worse refine), not always GT-best."
        ),
    }

    out = {
        "noise_t": args.noise_t,
        "noise_r": args.noise_r,
        "trials": args.trials,
        "methods": table,
        "recommendation": recommendation,
    }
    path = results_dir() / "handeye_bakeoff_summary.json"
    results_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
