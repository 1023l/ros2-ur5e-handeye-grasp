"""Hand-eye math: eye-to-hand solve (OpenCV) + YAML I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import yaml


@dataclass
class HandEyeResult:
    frame_id: str
    child_frame_id: str
    translation: Tuple[float, float, float]
    rotation_xyzw: Tuple[float, float, float, float]
    method: str = "Tsai"
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "child_frame_id": self.child_frame_id,
            "translation": [float(x) for x in self.translation],
            "rotation_xyzw": [float(x) for x in self.rotation_xyzw],
            "method": self.method,
            "note": self.note,
        }


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    return Rz @ Ry @ Rx


def R_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix → quaternion (x, y, z, w)."""
    m = np.asarray(R, dtype=float)
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def quat_xyzw_to_R(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def make_T(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def solve_eye_to_hand(
    T_base_ee_list: Sequence[np.ndarray],
    T_cam_board_list: Sequence[np.ndarray],
    method: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eye-to-hand: camera fixed in base, calibration board on end-effector.

    OpenCV eye-to-hand recipe: pass base←gripper (inverse of gripper2base);
    returned transform is cam→base. Prefer PARK/DANIILIDIS; TSAI is brittle here.
    """
    import cv2

    if method is None:
        method = cv2.CALIB_HAND_EYE_PARK

    R_b2g: List[np.ndarray] = []
    t_b2g: List[np.ndarray] = []
    R_t2c: List[np.ndarray] = []
    t_t2c: List[np.ndarray] = []

    for T_be, T_cb in zip(T_base_ee_list, T_cam_board_list):
        T_eb = inv_T(T_be)
        R_b2g.append(T_eb[:3, :3].copy())
        t_b2g.append(T_eb[:3, 3].reshape(3, 1).copy())
        R_t2c.append(T_cb[:3, :3].copy())
        t_t2c.append(T_cb[:3, 3].reshape(3, 1).copy())

    R_c2b, t_c2b = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=method)
    return np.asarray(R_c2b, dtype=float), np.asarray(t_c2b, dtype=float).reshape(3)


def save_handeye_yaml(path: Path, result: HandEyeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(result.as_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_handeye_yaml(path: Path) -> HandEyeResult:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tr = data["translation"]
    rq = data["rotation_xyzw"]
    return HandEyeResult(
        frame_id=str(data.get("frame_id", "base_link")),
        child_frame_id=str(data.get("child_frame_id", "arm_camera_optical_frame")),
        translation=(float(tr[0]), float(tr[1]), float(tr[2])),
        rotation_xyzw=(float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])),
        method=str(data.get("method", "")),
        note=str(data.get("note", "")),
    )


def synthesize_eye_to_hand_samples(
    T_base_cam_gt: np.ndarray,
    T_ee_board: np.ndarray,
    T_base_ee_list: Iterable[np.ndarray],
    noise_t: float = 0.0,
    noise_r_rad: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate (T_base_ee, T_cam_board) pairs from GT camera pose."""
    rng = rng or np.random.default_rng(0)
    T_cam_base = inv_T(T_base_cam_gt)
    outs_ee: List[np.ndarray] = []
    outs_cb: List[np.ndarray] = []
    for T_be in T_base_ee_list:
        T_bb = T_be @ T_ee_board
        T_cb = T_cam_base @ T_bb
        if noise_t > 0 or noise_r_rad > 0:
            dt = rng.normal(0.0, noise_t, size=3)
            dr = rng.normal(0.0, noise_r_rad, size=3)
            Rn = rpy_to_R(float(dr[0]), float(dr[1]), float(dr[2]))
            T_n = make_T(Rn, dt)
            T_cb = T_cb @ T_n
        outs_ee.append(np.asarray(T_be, dtype=float))
        outs_cb.append(np.asarray(T_cb, dtype=float))
    return outs_ee, outs_cb
