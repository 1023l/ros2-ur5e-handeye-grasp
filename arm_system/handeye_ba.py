"""Ceres-style nonlinear refinement for eye-to-hand (LM on pose residuals).

Per sample (board on EE):
  T_fk  = T_base_ee @ T_ee_board
  T_obs = T_base_cam @ T_cam_board
  r = [so3_log(R_fk^T R_obs),  t_obs - t_fk]

OpenCV provides init; this polish reports RMSE before/after under noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from arm_system.handeye_math import make_T


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v.reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    k = w / theta
    K = skew(k)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return (
            np.array(
                [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float
            )
            * 0.5
        )
    return (
        theta
        / (2.0 * math.sin(theta))
        * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float
        )
    )


def se3_exp(xi: np.ndarray) -> np.ndarray:
    w = xi[:3]
    v = xi[3:]
    R = so3_exp(w)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        V = np.eye(3)
    else:
        K = skew(w / theta)
        V = (
            np.eye(3)
            + (1 - math.cos(theta)) / theta * K
            + (theta - math.sin(theta)) / theta * (K @ K)
        )
    return make_T(R, V @ v)


@dataclass
class BAHandEyeReport:
    T_base_cam: np.ndarray
    rmse_init: float
    rmse_final: float
    iterations: int
    n_residuals: int
    converged: bool


def _stack_residuals(
    T_base_cam: np.ndarray,
    T_base_ee_list: Sequence[np.ndarray],
    T_cam_board_list: Sequence[np.ndarray],
    T_ee_board: np.ndarray,
) -> np.ndarray:
    chunks = []
    for A, B in zip(T_base_ee_list, T_cam_board_list):
        T_fk = A @ T_ee_board
        T_obs = T_base_cam @ B
        r_R = so3_log(T_fk[:3, :3].T @ T_obs[:3, :3])
        r_t = T_obs[:3, 3] - T_fk[:3, 3]
        chunks.append(np.concatenate([r_R, r_t]))
    return np.concatenate(chunks) if chunks else np.zeros(0)


def pose_residual_rmse(
    T_base_cam: np.ndarray,
    T_base_ee_list: Sequence[np.ndarray],
    T_cam_board_list: Sequence[np.ndarray],
    T_ee_board: np.ndarray,
) -> float:
    r = _stack_residuals(T_base_cam, T_base_ee_list, T_cam_board_list, T_ee_board)
    if r.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(r * r)))


def refine_eye_to_hand_ba(
    T_base_cam_init: np.ndarray,
    T_base_ee_list: Sequence[np.ndarray],
    T_cam_board_list: Sequence[np.ndarray],
    T_ee_board: np.ndarray,
    *,
    max_iters: int = 30,
    lambda0: float = 1e-2,
) -> BAHandEyeReport:
    T = np.array(T_base_cam_init, dtype=float, copy=True)
    rmse0 = pose_residual_rmse(T, T_base_ee_list, T_cam_board_list, T_ee_board)
    lam = lambda0
    last = rmse0
    it_used = 0
    converged = False
    eps = 1e-6

    for it in range(max_iters):
        it_used = it + 1
        r0 = _stack_residuals(T, T_base_ee_list, T_cam_board_list, T_ee_board)
        cols = []
        for k in range(6):
            xi = np.zeros(6)
            xi[k] = eps
            Tp = se3_exp(xi) @ T
            rk = _stack_residuals(Tp, T_base_ee_list, T_cam_board_list, T_ee_board)
            cols.append((rk - r0) / eps)
        J = np.column_stack(cols)
        H = J.T @ J + lam * np.eye(6)
        g = J.T @ r0
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        if float(np.linalg.norm(delta)) < 1e-12:
            converged = True
            break
        T_new = se3_exp(delta) @ T
        rmse_new = pose_residual_rmse(
            T_new, T_base_ee_list, T_cam_board_list, T_ee_board
        )
        if rmse_new < last - 1e-15:
            T = T_new
            last = rmse_new
            lam = max(lam * 0.4, 1e-8)
            if float(np.linalg.norm(delta)) < 1e-9:
                converged = True
                break
        else:
            lam *= 6.0

    return BAHandEyeReport(
        T_base_cam=T,
        rmse_init=rmse0,
        rmse_final=last,
        iterations=it_used,
        n_residuals=6 * len(T_base_ee_list),
        converged=converged or last <= rmse0,
    )
