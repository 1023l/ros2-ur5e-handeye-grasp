#!/usr/bin/env python3
"""Stress sweep: raise occlusion / hand-eye noise until metrics cliff.

'Break on purpose' = soak / HALT style load until the weak link shows,
then tune. Not sabotage.

  python3 scripts/eval_stress.py
  ros2 run arm_system eval_stress.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for p in (str(_PKG), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from arm_system.handeye_ba import refine_eye_to_hand_ba
from arm_system.handeye_ceres_style import (
    refine_handeye_consistency_lm,
    synthesize_consistency_samples_eye_to_hand,
)
from arm_system.handeye_math import make_T, rpy_to_R, solve_eye_to_hand, synthesize_eye_to_hand_samples
from arm_system.metrics import MetricsLogger
from eval_vision_grasp import run_trial


def results_dir() -> Path:
    return _PKG / "results"


def _vision_at(occlusion: float, trials: int) -> Dict[str, Any]:
    logger = MetricsLogger(results_dir(), run_name=f"stress_vis_occ{int(occlusion * 100):02d}")
    for i in range(trials):
        rec = run_trial(
            i,
            target="red",
            occlusion_prob=occlusion,
            object_size_m=0.06,
            fx=600.0,
            fy=600.0,
            use_apparent=True,
        )
        logger.log(rec)
    s = logger.summarize()
    bad = sum(1 for r in logger.records if r.failure_mode == "pose_inaccurate")
    s["pose_ok_rate"] = 1.0 - bad / max(len(logger.records), 1)
    s["occlusion"] = occlusion
    return s


def _handeye_at(noise_t: float, noise_r: float, trials: int) -> Dict[str, Any]:
    R_gt = rpy_to_R(math.pi, 0.0, 0.0)
    t_gt = np.array([0.40, 0.0, 0.50])
    T_gt = make_T(R_gt, t_gt)
    T_ee_board = make_T(np.eye(3), [0.0, 0.0, 0.08])
    err_cv, err_ba, err_ceres, cons0, cons1 = [], [], [], [], []
    for i in range(trials):
        rng = np.random.default_rng(300 + i)
        T_ees = []
        for _ in range(10):
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
            T_gt, T_ee_board, T_ees, noise_t=noise_t, noise_r_rad=noise_r, rng=rng
        )
        R_cv, t_cv = solve_eye_to_hand(T_be, T_cb)
        T_cv = make_T(R_cv, t_cv)
        ba = refine_eye_to_hand_ba(T_cv, T_be, T_cb, T_ee_board)
        samples = synthesize_consistency_samples_eye_to_hand(
            T_gt, T_ee_board, T_ees, noise_t=noise_t, noise_r=noise_r, rng=rng
        )
        T_ref, r0, r1 = refine_handeye_consistency_lm(T_cv, samples, eye_to_hand=True)
        err_cv.append(float(np.linalg.norm(t_cv - t_gt)))
        err_ba.append(float(np.linalg.norm(ba.T_base_cam[:3, 3] - t_gt)))
        err_ceres.append(float(np.linalg.norm(T_ref[:3, 3] - t_gt)))
        cons0.append(r0.rmse_combined)
        cons1.append(r1.rmse_combined)
    return {
        "noise_t": noise_t,
        "noise_r": noise_r,
        "mean_err_cv": sum(err_cv) / len(err_cv),
        "mean_err_ba": sum(err_ba) / len(err_ba),
        "mean_err_ceres_gate": sum(err_ceres) / len(err_ceres),
        "mean_cons_before": sum(cons0) / len(cons0),
        "mean_cons_after": sum(cons1) / len(cons1),
        "ba_helps": sum(1 for a, b in zip(err_ba, err_cv) if a <= b + 1e-9) / len(err_cv),
        "ceres_not_worse": sum(1 for a, b in zip(err_ceres, err_cv) if a <= b + 1e-9)
        / len(err_cv),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stress sweep occlusion & hand-eye noise")
    p.add_argument("--vision-trials", type=int, default=30)
    p.add_argument("--handeye-trials", type=int, default=12)
    args = p.parse_args(argv)

    vision_rows: List[Dict[str, Any]] = []
    for occ in (0.0, 0.40, 0.55, 0.75, 0.90):
        vision_rows.append(_vision_at(occ, args.vision_trials))

    handeye_rows: List[Dict[str, Any]] = []
    for nt, nr in ((0.002, 0.01), (0.004, 0.02), (0.008, 0.04), (0.015, 0.08)):
        handeye_rows.append(_handeye_at(nt, nr, args.handeye_trials))

    cliff_occ = None
    for row in vision_rows:
        if row.get("detect_rate", 1) < 0.7 or row.get("pose_ok_rate", 1) < 0.7:
            cliff_occ = row["occlusion"]
            break

    out = {
        "vision_sweep": vision_rows,
        "handeye_sweep": handeye_rows,
        "vision_cliff_occlusion": cliff_occ,
        "note": "Raise load until cliff; tune detector/depth/handeye gate.",
    }
    path = results_dir() / "stress_sweep_summary.json"
    results_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
