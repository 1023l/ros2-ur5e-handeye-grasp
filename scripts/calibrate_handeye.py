#!/usr/bin/env python3
"""Calibrate eye-to-hand and write config/handeye_result.yaml.

Default mode is synthetic (no robot): recover a known camera pose with
OpenCV calibrateHandEye to prove the solver path. Optional --live moves
the UR via MoveIt and uses TF tool0 poses + simulated board observations
with the same ground-truth camera model (still no physical Charuco).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


def _default_out() -> Path:
    # Prefer installed share path; fall back to source tree config/
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("arm_system"))
        return share / "config" / "handeye_result.yaml"
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[1] / "config" / "handeye_result.yaml"


def _source_out() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "handeye_result.yaml"


def run_synthetic(out: Path, samples: int, noise_t: float, noise_r: float, use_ba: bool) -> int:
    from arm_system.handeye_ba import refine_eye_to_hand_ba
    from arm_system.handeye_math import (
        HandEyeResult,
        R_to_quat_xyzw,
        make_T,
        rpy_to_R,
        save_handeye_yaml,
        solve_eye_to_hand,
        synthesize_eye_to_hand_samples,
    )

    # Same GT as previous demo static TF: above table, looking down.
    R_gt = rpy_to_R(math.pi, 0.0, 0.0)
    t_gt = np.array([0.40, 0.0, 0.50], dtype=float)
    T_base_cam = make_T(R_gt, t_gt)
    T_ee_board = make_T(np.eye(3), [0.0, 0.0, 0.08])

    T_ees = []
    rng = np.random.default_rng(7)
    for _ in range(samples):
        yaw = float(rng.uniform(-0.8, 0.8))
        pitch = float(rng.uniform(-0.4, 0.2))
        roll = float(rng.uniform(-0.3, 0.3))
        R = rpy_to_R(roll, pitch, yaw)
        t = np.array(
            [
                float(rng.uniform(0.25, 0.55)),
                float(rng.uniform(-0.25, 0.25)),
                float(rng.uniform(0.15, 0.45)),
            ]
        )
        T_ees.append(make_T(R, t))

    T_be, T_cb = synthesize_eye_to_hand_samples(
        T_base_cam, T_ee_board, T_ees, noise_t=noise_t, noise_r_rad=noise_r, rng=rng
    )
    R_est, t_est = solve_eye_to_hand(T_be, T_cb)
    T_cv = make_T(R_est, t_est)
    err_t_cv = float(np.linalg.norm(t_est - t_gt))
    R_err = R_est @ R_gt.T
    ang_cv = math.acos(float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)))

    method = "Park-eye-to-hand"
    note = f"opencv err_t={err_t_cv:.6f}m err_r={ang_cv:.6f}rad"
    T_final = T_cv
    ba_note = ""

    if use_ba:
        report = refine_eye_to_hand_ba(T_cv, T_be, T_cb, T_ee_board)
        T_final = report.T_base_cam
        t_ba = T_final[:3, 3]
        R_ba = T_final[:3, :3]
        err_t_ba = float(np.linalg.norm(t_ba - t_gt))
        ang_ba = math.acos(
            float(np.clip((np.trace(R_ba @ R_gt.T) - 1.0) * 0.5, -1.0, 1.0))
        )
        method = "Park+SE3-LM-BA"
        ba_note = (
            f" ba_rmse {report.rmse_init:.6f}->{report.rmse_final:.6f}"
            f" iters={report.iterations}"
            f" err_t={err_t_ba:.6f}m err_r={ang_ba:.6f}rad"
        )
        note = note + ba_note
        t_est = t_ba
        R_est = R_ba
        err_t_cv = err_t_ba
        ang_cv = ang_ba

    q = R_to_quat_xyzw(R_est)
    result = HandEyeResult(
        frame_id="base_link",
        child_frame_id="arm_camera_optical_frame",
        translation=(float(t_est[0]), float(t_est[1]), float(t_est[2])),
        rotation_xyzw=q,
        method=method,
        note=(f"synthetic samples={samples} " + note),
    )
    save_handeye_yaml(out, result)
    src = _source_out()
    if src.resolve() != out.resolve():
        save_handeye_yaml(src, result)

    print(f"Wrote {out}")
    print(f"method {method}")
    print(f"translation {result.translation}")
    print(f"GT translation {tuple(t_gt.tolist())}")
    print(f"error translation {err_t_cv:.6f} m, rotation {ang_cv:.6f} rad")
    if ba_note:
        print(ba_note.strip())
    return 0 if err_t_cv < 0.05 and ang_cv < 0.1 else 1


def run_live(out: Path, samples: int) -> int:
    """Move arm to named/random joints; build board obs from GT camera model + TF."""
    import rclpy
    from tf2_ros import Buffer, TransformListener
    from rclpy.duration import Duration

    from arm_system.handeye_math import (
        HandEyeResult,
        R_to_quat_xyzw,
        make_T,
        quat_xyzw_to_R,
        rpy_to_R,
        save_handeye_yaml,
        solve_eye_to_hand,
        synthesize_eye_to_hand_samples,
    )
    from arm_system.moveit_client import MoveItArmClient, NAMED_JOINT_TARGETS, ARM_JOINTS

    rclpy.init()
    arm = MoveItArmClient(node_name="handeye_calib_arm")
    tf_node = arm  # reuse
    buffer = Buffer()
    listener = TransformListener(buffer, arm)

    if not arm.wait_ready(15.0):
        arm.destroy_node()
        rclpy.shutdown()
        return 2

    # Wait TF tool0
    import time

    t0 = time.time()
    while time.time() - t0 < 10.0:
        try:
            buffer.lookup_transform("base_link", "tool0", rclpy.time.Time(), timeout=Duration(seconds=0.2))
            break
        except Exception:  # noqa: BLE001
            rclpy.spin_once(arm, timeout_sec=0.1)
    else:
        arm.get_logger().error("No TF base_link→tool0")
        arm.destroy_node()
        rclpy.shutdown()
        return 2

    R_gt = rpy_to_R(math.pi, 0.0, 0.0)
    t_gt = np.array([0.40, 0.0, 0.50], dtype=float)
    T_base_cam = make_T(R_gt, t_gt)
    T_ee_board = make_T(np.eye(3), [0.0, 0.0, 0.08])

    # Pose seeds: home/up + a few joint nudges
    seeds = [NAMED_JOINT_TARGETS["home"], NAMED_JOINT_TARGETS["up"]]
    rng = np.random.default_rng(3)
    while len(seeds) < samples:
        j = dict(NAMED_JOINT_TARGETS["up"])
        j["shoulder_pan_joint"] += float(rng.uniform(-0.5, 0.5))
        j["elbow_joint"] += float(rng.uniform(-0.4, 0.4))
        j["wrist_1_joint"] += float(rng.uniform(-0.4, 0.4))
        seeds.append(j)

    T_ees = []
    for i, joints in enumerate(seeds[:samples]):
        arm.get_logger().info(f"Calib move {i+1}/{samples}")
        code = arm.move_joints(joints, velocity_scale=0.25, accel_scale=0.25)
        if code != 1:
            arm.get_logger().warn(f"Skip failed move {i}")
            continue
        # settle + lookup
        for _ in range(10):
            rclpy.spin_once(arm, timeout_sec=0.05)
        try:
            tf = buffer.lookup_transform(
                "base_link", "tool0", rclpy.time.Time(), timeout=Duration(seconds=1.0)
            )
        except Exception as exc:  # noqa: BLE001
            arm.get_logger().warn(f"TF fail: {exc}")
            continue
        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_xyzw_to_R([q.x, q.y, q.z, q.w])
        T_ees.append(make_T(R, [t.x, t.y, t.z]))

    if len(T_ees) < 4:
        arm.get_logger().error(f"Need >=4 samples, got {len(T_ees)}")
        arm.destroy_node()
        rclpy.shutdown()
        return 1

    T_be, T_cb = synthesize_eye_to_hand_samples(T_base_cam, T_ee_board, T_ees)
    R_est, t_est = solve_eye_to_hand(T_be, T_cb)
    err_t = float(np.linalg.norm(t_est - t_gt))
    q = R_to_quat_xyzw(R_est)
    result = HandEyeResult(
        frame_id="base_link",
        child_frame_id="arm_camera_optical_frame",
        translation=(float(t_est[0]), float(t_est[1]), float(t_est[2])),
        rotation_xyzw=q,
        method="Park-eye-to-hand-liveTF",
        note=f"live samples={len(T_ees)} err_t={err_t:.6f}m",
    )
    save_handeye_yaml(out, result)
    save_handeye_yaml(_source_out(), result)
    arm.get_logger().info(f"Wrote {out} err_t={err_t:.6f}")
    arm.destroy_node()
    rclpy.shutdown()
    return 0 if err_t < 0.05 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Eye-to-hand calibration → YAML")
    p.add_argument("--mode", choices=("synthetic", "live"), default="synthetic")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--noise-t", type=float, default=0.0, help="Synthetic translation noise (m)")
    p.add_argument("--noise-r", type=float, default=0.0, help="Synthetic rotation noise (rad)")
    p.add_argument("--no-ba", action="store_true", help="Skip SE3-LM BA polish")
    p.add_argument("--out", type=Path, default=None, help="Output YAML path")
    args = p.parse_args(argv)
    out = args.out or _source_out()
    if args.mode == "synthetic":
        return run_synthetic(
            out, args.samples, args.noise_t, args.noise_r, use_ba=not args.no_ba
        )
    return run_live(out, args.samples)


if __name__ == "__main__":
    sys.exit(main())
