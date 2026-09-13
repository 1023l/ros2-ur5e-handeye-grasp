#!/usr/bin/env python3
"""Eval visionguide-style Ceres pair-consistency residual (eye-to-hand).

Compares OpenCV hand-eye init vs consistency-LM polish — same residual family
as produce/CeresEyeToHandProblem.h OptBaseInCamCeres.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from arm_system.handeye_ceres_style import (
    evaluate_consistency,
    refine_handeye_consistency_lm,
    synthesize_consistency_samples_eye_to_hand,
)
from arm_system.handeye_math import make_T, rpy_to_R, solve_eye_to_hand, synthesize_eye_to_hand_samples
from arm_system.metrics import MetricsLogger, TrialRecord


def results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--noise-t", type=float, default=0.003)
    p.add_argument("--noise-r", type=float, default=0.015)
    args = p.parse_args(argv)

    R_gt = rpy_to_R(math.pi, 0.0, 0.0)
    t_gt = np.array([0.40, 0.0, 0.50])
    T_cam = make_T(R_gt, t_gt)
    T_ee_board = make_T(np.eye(3), [0.0, 0.0, 0.08])
    logger = MetricsLogger(results_dir(), run_name="handeye_ceres_style")

    for i in range(args.trials):
        rng = np.random.default_rng(200 + i)
        T_tools = []
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
            T_tools.append(make_T(R, t))

        # OpenCV path uses T_base_ee + T_cam_board
        T_be, T_cb = synthesize_eye_to_hand_samples(
            T_cam, T_ee_board, T_tools, noise_t=args.noise_t, noise_r_rad=args.noise_r, rng=rng
        )
        R_cv, t_cv = solve_eye_to_hand(T_be, T_cb)
        T_cv = make_T(R_cv, t_cv)

        samples = synthesize_consistency_samples_eye_to_hand(
            T_cam,
            T_ee_board,
            T_tools,
            noise_t=args.noise_t,
            noise_r=args.noise_r,
            rng=rng,
        )
        rep_cv = evaluate_consistency(T_cv, samples, eye_to_hand=True)
        T_ref, rep0, rep1 = refine_handeye_consistency_lm(
            T_cv, samples, eye_to_hand=True
        )
        err_cv = float(np.linalg.norm(t_cv - t_gt))
        err_ref = float(np.linalg.norm(T_ref[:3, 3] - t_gt))

        logger.log(
            TrialRecord(
                trial_id=i,
                timestamp=time.time(),
                scene_mode="ceres_style",
                target_label="eye_to_hand",
                detected=True,
                depth_error_m=err_ref,
                pose_method="consistency_lm",
                notes=(
                    f"cv_err={err_cv:.5f} ref_err={err_ref:.5f} "
                    f"cons_rmse {rep_cv.rmse_combined:.5f}->{rep1.rmse_combined:.5f}"
                ),
                extra={
                    "err_t_opencv": err_cv,
                    "err_t_refined": err_ref,
                    "cons_rmse_opencv": rep_cv.rmse_combined,
                    "cons_rmse_refined": rep1.rmse_combined,
                    "cons_rmse_init_at_cv": rep0.rmse_combined,
                    "improved_err": err_ref <= err_cv + 1e-9,
                    "improved_cons": rep1.rmse_combined <= rep_cv.rmse_combined + 1e-12,
                },
            )
        )

    rows = logger.records
    summary = logger.summarize()
    summary.update(
        {
            "mean_err_t_opencv": sum(r.extra["err_t_opencv"] for r in rows) / len(rows),
            "mean_err_t_refined": sum(r.extra["err_t_refined"] for r in rows) / len(rows),
            "mean_cons_rmse_opencv": sum(r.extra["cons_rmse_opencv"] for r in rows) / len(rows),
            "mean_cons_rmse_refined": sum(r.extra["cons_rmse_refined"] for r in rows) / len(rows),
            "note": "Residual family matches visionguide CeresEyeToHand OptBaseInCamCeres",
        }
    )
    logger.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
