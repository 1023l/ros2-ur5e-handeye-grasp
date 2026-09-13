#!/usr/bin/env python3
"""Compare OpenCV hand-eye vs +SE3-LM BA under observation noise (no robot)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from arm_system.handeye_ba import pose_residual_rmse, refine_eye_to_hand_ba
from arm_system.handeye_math import (
    make_T,
    rpy_to_R,
    solve_eye_to_hand,
    synthesize_eye_to_hand_samples,
)
from arm_system.metrics import MetricsLogger, TrialRecord


def results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--noise-t", type=float, default=0.004)
    p.add_argument("--noise-r", type=float, default=0.02)
    args = p.parse_args(argv)

    R_gt = rpy_to_R(math.pi, 0.0, 0.0)
    t_gt = np.array([0.40, 0.0, 0.50])
    T_gt = make_T(R_gt, t_gt)
    T_ee_board = make_T(np.eye(3), [0.0, 0.0, 0.08])
    logger = MetricsLogger(results_dir(), run_name="handeye_ba")

    for i in range(args.trials):
        rng = np.random.default_rng(100 + i)
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
        R_cv, t_cv = solve_eye_to_hand(T_be, T_cb)
        T_cv = make_T(R_cv, t_cv)
        report = refine_eye_to_hand_ba(T_cv, T_be, T_cb, T_ee_board)
        err_cv = float(np.linalg.norm(t_cv - t_gt))
        err_ba = float(np.linalg.norm(report.T_base_cam[:3, 3] - t_gt))
        logger.log(
            TrialRecord(
                trial_id=i,
                timestamp=time.time(),
                scene_mode="handeye",
                target_label="cam",
                detected=True,
                depth_error_m=err_ba,
                pose_method="ba",
                notes=f"cv_err={err_cv:.5f} ba_err={err_ba:.5f} "
                f"rmse {report.rmse_init:.5f}->{report.rmse_final:.5f}",
                extra={
                    "err_t_opencv": err_cv,
                    "err_t_ba": err_ba,
                    "rmse_init": report.rmse_init,
                    "rmse_final": report.rmse_final,
                    "improved": err_ba <= err_cv + 1e-6,
                },
            )
        )

    # Aggregate
    rows = logger.records
    mean_cv = sum(r.extra["err_t_opencv"] for r in rows) / len(rows)
    mean_ba = sum(r.extra["err_t_ba"] for r in rows) / len(rows)
    improved = sum(1 for r in rows if r.extra["improved"]) / len(rows)
    summary = logger.summarize()
    summary.update(
        {
            "mean_err_t_opencv": mean_cv,
            "mean_err_t_ba": mean_ba,
            "ba_better_or_equal_rate": improved,
            "noise_t": args.noise_t,
            "noise_r": args.noise_r,
        }
    )
    logger.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if mean_ba <= mean_cv + 0.002 else 1


if __name__ == "__main__":
    sys.exit(main())
