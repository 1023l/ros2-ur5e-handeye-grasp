"""Ceres-style residuals aligned with visionguide produce/CeresEye*Problem.h

Your industrial stack (visionguide) uses Google Ceres for:
  1) PnP reprojection (3D board point → image uv residual)
  2) Multi-pose hand-eye consistency:
       eye-to-hand: objInTool = BaseInTool * camInBase * objInCam
       pair residual: pose_i - pose_j  (object fixed → poses agree)

This module reimplements (2) in pure NumPy for ROS demos / metrics, so the
arm_system eval speaks the same language as your KUKA/visionguide calib —
without linking the proprietary Qt/Halcon binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from arm_system.handeye_ba import so3_exp, so3_log
from arm_system.handeye_math import make_T, rpy_to_R


def aa_t_to_T(rx: float, ry: float, rz: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """Angle-axis (rad) + translation → 4x4 (matches ceres AngleAxis convention)."""
    R = so3_exp(np.array([rx, ry, rz], dtype=float))
    return make_T(R, [tx, ty, tz])


def T_to_aa_t(T: np.ndarray) -> np.ndarray:
    w = so3_log(T[:3, :3])
    t = T[:3, 3]
    return np.concatenate([w, t])


def compose_obj_in_tool_eye_to_hand(
    T_base_in_tool: np.ndarray,
    T_cam_in_base: np.ndarray,
    T_obj_in_cam: np.ndarray,
) -> np.ndarray:
    """visionguide calHandEyePose (eye-to-hand): objInTool = BaseInTool * camInBase * objInCam."""
    return T_base_in_tool @ T_cam_in_base @ T_obj_in_cam


def compose_obj_in_base_eye_in_hand(
    T_tool_in_base: np.ndarray,
    T_cam_in_tool: np.ndarray,
    T_obj_in_cam: np.ndarray,
) -> np.ndarray:
    """visionguide CalAHB_pose (eye-in-hand): objInBase = ToolInBase * camInTool * objInCam."""
    return T_tool_in_base @ T_cam_in_tool @ T_obj_in_cam


@dataclass
class ConsistencySample:
    """One calib snapshot (robot pose + object pose in camera)."""
    # For eye-to-hand store BaseInTool; for eye-in-hand store ToolInBase.
    T_robot: np.ndarray
    T_obj_in_cam: np.ndarray


@dataclass
class ConsistencyReport:
    mode: str  # eye_to_hand | eye_in_hand
    n_pairs: int
    rmse_rot: float
    rmse_trans: float
    rmse_combined: float


def pair_consistency_residuals(
    T_handeye: np.ndarray,
    samples: Sequence[ConsistencySample],
    *,
    eye_to_hand: bool = True,
    w_rot: float = 1.0,
    w_trans: float = 1.0,
) -> np.ndarray:
    """
    Build stacked 6D residuals over all sample pairs (i<j), same idea as
    OptBaseInCamCeres / OptCameraCeres in visionguide.
    """
    chunks: List[np.ndarray] = []
    n = len(samples)
    for i in range(n):
        for j in range(i + 1, n):
            if eye_to_hand:
                p0 = compose_obj_in_tool_eye_to_hand(
                    samples[i].T_robot, T_handeye, samples[i].T_obj_in_cam
                )
                p1 = compose_obj_in_tool_eye_to_hand(
                    samples[j].T_robot, T_handeye, samples[j].T_obj_in_cam
                )
            else:
                p0 = compose_obj_in_base_eye_in_hand(
                    samples[i].T_robot, T_handeye, samples[i].T_obj_in_cam
                )
                p1 = compose_obj_in_base_eye_in_hand(
                    samples[j].T_robot, T_handeye, samples[j].T_obj_in_cam
                )
            dR = so3_log(p0[:3, :3].T @ p1[:3, :3])
            dt = p1[:3, 3] - p0[:3, 3]
            chunks.append(np.concatenate([w_rot * dR, w_trans * dt]))
    if not chunks:
        return np.zeros(0)
    return np.concatenate(chunks)


def evaluate_consistency(
    T_handeye: np.ndarray,
    samples: Sequence[ConsistencySample],
    *,
    eye_to_hand: bool = True,
) -> ConsistencyReport:
    r = pair_consistency_residuals(T_handeye, samples, eye_to_hand=eye_to_hand)
    n_pairs = len(samples) * (len(samples) - 1) // 2
    if r.size == 0:
        return ConsistencyReport(
            mode="eye_to_hand" if eye_to_hand else "eye_in_hand",
            n_pairs=0,
            rmse_rot=0.0,
            rmse_trans=0.0,
            rmse_combined=0.0,
        )
    # r is [rot3, trans3] * n_pairs
    r = r.reshape(-1, 6)
    rmse_rot = float(np.sqrt(np.mean(r[:, :3] ** 2)))
    rmse_trans = float(np.sqrt(np.mean(r[:, 3:] ** 2)))
    rmse_comb = float(np.sqrt(np.mean(r ** 2)))
    return ConsistencyReport(
        mode="eye_to_hand" if eye_to_hand else "eye_in_hand",
        n_pairs=n_pairs,
        rmse_rot=rmse_rot,
        rmse_trans=rmse_trans,
        rmse_combined=rmse_comb,
    )


def refine_handeye_consistency_lm(
    T_handeye_init: np.ndarray,
    samples: Sequence[ConsistencySample],
    *,
    eye_to_hand: bool = True,
    max_iters: int = 25,
    lambda0: float = 1e-2,
    w_rot: float = 1.0,
    w_trans: float = 2.5,
    w_prior: float = 0.35,
    max_trans_drift_m: float = 0.03,
) -> Tuple[np.ndarray, ConsistencyReport, ConsistencyReport]:
    """LM polish with soft prior toward OpenCV init (anti-overfit under noise).

    Accept refined T only if consistency improves and translation drift from
    init stays within max_trans_drift_m — otherwise keep init.
    """
    from arm_system.handeye_ba import se3_exp, so3_log as _so3_log

    T_init = np.array(T_handeye_init, dtype=float, copy=True)
    T = T_init.copy()
    rep0 = evaluate_consistency(T, samples, eye_to_hand=eye_to_hand)
    lam = lambda0
    last = rep0.rmse_combined
    eps = 1e-6

    def residuals(Tm: np.ndarray) -> np.ndarray:
        r = pair_consistency_residuals(
            Tm, samples, eye_to_hand=eye_to_hand, w_rot=w_rot, w_trans=w_trans
        )
        # Soft SE3 prior toward init (angle-axis + translation)
        dR = _so3_log(T_init[:3, :3].T @ Tm[:3, :3])
        dt = Tm[:3, 3] - T_init[:3, 3]
        prior = w_prior * np.concatenate([dR, dt])
        if r.size == 0:
            return prior
        return np.concatenate([r, prior])

    for _ in range(max_iters):
        r0 = residuals(T)
        if r0.size == 0:
            break
        cols = []
        for k in range(6):
            xi = np.zeros(6)
            xi[k] = eps
            Tp = se3_exp(xi) @ T
            rk = residuals(Tp)
            cols.append((rk - r0) / eps)
        J = np.column_stack(cols)
        H = J.T @ J + lam * np.eye(6)
        g = J.T @ r0
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        T_new = se3_exp(delta) @ T
        # Use consistency (not prior-augmented) for line search
        rep_new = evaluate_consistency(T_new, samples, eye_to_hand=eye_to_hand)
        if rep_new.rmse_combined < last:
            T = T_new
            last = rep_new.rmse_combined
            lam = max(lam * 0.4, 1e-8)
            if float(np.linalg.norm(delta)) < 1e-9:
                break
        else:
            lam *= 6.0

    rep1 = evaluate_consistency(T, samples, eye_to_hand=eye_to_hand)
    drift = float(np.linalg.norm(T[:3, 3] - T_init[:3, 3]))
    # Gate: don't ship a refine that wanders or fails to help consistency
    if rep1.rmse_combined > rep0.rmse_combined - 1e-12 or drift > max_trans_drift_m:
        return T_init, rep0, rep0
    return T, rep0, rep1


def synthesize_consistency_samples_eye_to_hand(
    T_cam_in_base_gt: np.ndarray,
    T_ee_board: np.ndarray,
    T_tool_in_base_list: Sequence[np.ndarray],
    *,
    noise_t: float = 0.0,
    noise_r: float = 0.0,
    rng: np.random.Generator | None = None,
) -> List[ConsistencySample]:
    """
    Build samples as visionguide would see them for eye-to-hand:
      robot stores BaseInTool = inv(ToolInBase)
      objInCam = inv(camInBase) * ToolInBase * ee_board
    """
    rng = rng or np.random.default_rng(0)
    out: List[ConsistencySample] = []
    for T_tb in T_tool_in_base_list:
        T_base_in_tool = np.linalg.inv(T_tb)
        T_obj_in_cam = np.linalg.inv(T_cam_in_base_gt) @ T_tb @ T_ee_board
        if noise_t > 0 or noise_r > 0:
            dR = so3_exp(rng.normal(0.0, noise_r, size=3))
            dt = rng.normal(0.0, noise_t, size=3)
            T_obj_in_cam = make_T(dR, dt) @ T_obj_in_cam
        out.append(ConsistencySample(T_robot=T_base_in_tool, T_obj_in_cam=T_obj_in_cam))
    return out
