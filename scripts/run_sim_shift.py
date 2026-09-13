#!/usr/bin/env python3
"""One 'sim factory shift' — portfolio-grade campaign without real hardware.

Mirrors industrial workflow language (calib → vision → grasp ranking → optional
planning), writes a single shift report under results/shifts/.

  # Offline only (no MoveIt / no robot):
  python3 scripts/run_sim_shift.py --profile quick

  # After MoveIt fake is up:
  ros2 run arm_system run_sim_shift.py --profile full --with-planning
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure package import when run as installed script or via PYTHONPATH
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from arm_system.handeye_ceres_style import (
    refine_handeye_consistency_lm,
    synthesize_consistency_samples_eye_to_hand,
)
from arm_system.handeye_math import make_T, rpy_to_R, solve_eye_to_hand, synthesize_eye_to_hand_samples


def results_root() -> Path:
    return _PKG_ROOT / "results"


def shift_dir(name: str) -> Path:
    d = results_root() / "shifts" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_visionguide_style_session(
    out_path: Path,
    *,
    n_poses: int = 12,
    noise_t: float = 0.003,
    noise_r: float = 0.015,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Synthetic eye-to-hand session in a JSON schema close to industrial dumps:
      samples[i].T_base_in_tool (4x4 row-major)
      samples[i].T_obj_in_cam
    Same composition as visionguide calHandEyePose.
    """
    rng = np.random.default_rng(seed)
    R_gt = rpy_to_R(math.pi, 0.0, 0.0)
    t_gt = np.array([0.40, 0.0, 0.50])
    T_cam = make_T(R_gt, t_gt)
    T_ee_board = make_T(np.eye(3), [0.0, 0.0, 0.08])
    T_tools: List[np.ndarray] = []
    for _ in range(n_poses):
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

    samples = synthesize_consistency_samples_eye_to_hand(
        T_cam, T_ee_board, T_tools, noise_t=noise_t, noise_r=noise_r, rng=rng
    )
    T_be, T_cb = synthesize_eye_to_hand_samples(
        T_cam, T_ee_board, T_tools, noise_t=noise_t, noise_r_rad=noise_r, rng=rng
    )
    R_cv, t_cv = solve_eye_to_hand(T_be, T_cb)
    T_cv = make_T(R_cv, t_cv)
    T_ref, rep0, rep1 = refine_handeye_consistency_lm(T_cv, samples, eye_to_hand=True)

    payload = {
        "schema": "arm_system.handeye_session.v1",
        "mode": "eye_to_hand",
        "note": "Synthetic session; residual family aligned with visionguide CeresEyeToHand",
        "noise_t_m": noise_t,
        "noise_r_rad": noise_r,
        "gt_cam_in_base": T_cam.tolist(),
        "opencv_cam_in_base": T_cv.tolist(),
        "refined_cam_in_base": T_ref.tolist(),
        "consistency_rmse_opencv": rep0.rmse_combined,
        "consistency_rmse_refined": rep1.rmse_combined,
        "err_t_opencv_m": float(np.linalg.norm(t_cv - t_gt)),
        "err_t_refined_m": float(np.linalg.norm(T_ref[:3, 3] - t_gt)),
        "samples": [
            {
                "T_base_in_tool": s.T_robot.tolist(),
                "T_obj_in_cam": s.T_obj_in_cam.tolist(),
            }
            for s in samples
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the outermost JSON object from stdout (handles nested braces)."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def run_module(script_name: str, argv: List[str], env_pythonpath: str) -> Dict[str, Any]:
    script = _PKG_ROOT / "scripts" / script_name
    cmd = [sys.executable, str(script), *argv]
    env = os.environ.copy()
    env["PYTHONPATH"] = env_pythonpath
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(_PKG_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    out: Dict[str, Any] = {
        "script": script_name,
        "argv": argv,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 2),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }
    summary = _extract_json_object(proc.stdout or "")
    if summary is not None:
        out["summary"] = summary
    return out


def write_report(path: Path, meta: Dict[str, Any], stages: List[Dict[str, Any]]) -> None:
    lines = [
        f"# Sim Factory Shift — {meta['shift_id']}",
        "",
        f"- profile: `{meta['profile']}`",
        f"- started: {meta['started']}",
        f"- finished: {meta['finished']}",
        f"- with_planning: {meta['with_planning']}",
        "",
        "## Pipeline",
        "",
        "UR5e + MoveIt fake + hard-scene vision + industrial-style hand-eye residuals:",
        "noise injection → OpenCV init → consistency refine → grasp scoring → optional obstacle planning.",
        "Metrics: consistency RMSE, detect rate, depth error, plan_rate, failure modes.",
        "",
        "## Stages",
        "",
    ]
    for s in stages:
        ok = "OK" if s.get("returncode", 1) == 0 else "FAIL"
        lines.append(f"### {s.get('name', s.get('script'))} — {ok} ({s.get('elapsed_s')}s)")
        summ = s.get("summary")
        if isinstance(summ, dict):
            for k in (
                "mean_err_t_opencv",
                "mean_err_t_ba",
                "mean_err_t_refined",
                "mean_cons_rmse_opencv",
                "mean_cons_rmse_refined",
                "detect_rate",
                "mean_pixel_error",
                "mean_depth_error_m",
                "plan_success_rate",
                "vision_cliff_occlusion",
                "n",
            ):
                if k in summ and summ[k] is not None:
                    lines.append(f"- **{k}**: {summ[k]}")
            if "vision_sweep" in summ and isinstance(summ["vision_sweep"], list):
                last = summ["vision_sweep"][-1] if summ["vision_sweep"] else {}
                lines.append(
                    f"- stress_vis_last: detect={last.get('detect_rate')} "
                    f"pose_ok={last.get('pose_ok_rate')} occ={last.get('occlusion')}"
                )
        if s.get("artifact"):
            lines.append(f"- artifact: `{s['artifact']}`")
        if s.get("returncode", 0) != 0 and s.get("stderr_tail"):
            lines.append(f"- stderr: `{s['stderr_tail'][:200]}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run one sim-as-real factory shift")
    p.add_argument("--profile", choices=("quick", "full"), default="quick")
    p.add_argument("--with-planning", action="store_true", help="Requires MoveIt fake running")
    p.add_argument("--shift-id", default="", help="Default: timestamp")
    args = p.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    shift_id = args.shift_id or f"shift_{stamp}"
    out = shift_dir(shift_id)
    py_path = str(_PKG_ROOT)
    if os.environ.get("PYTHONPATH"):
        py_path = py_path + os.pathsep + os.environ["PYTHONPATH"]

    if args.profile == "quick":
        he_trials, he_noise_t, he_noise_r = 8, 0.004, 0.02
        vision_trials = 20
        plan_trials = 6
        n_session = 10
    else:
        he_trials, he_noise_t, he_noise_r = 20, 0.004, 0.02
        vision_trials = 50
        plan_trials = 12
        n_session = 12

    meta = {
        "shift_id": shift_id,
        "profile": args.profile,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "with_planning": bool(args.with_planning),
        "finished": "",
    }
    stages: List[Dict[str, Any]] = []

    # 1) Industrial-style session dump (always)
    session_path = out / "handeye_session.json"
    session = export_visionguide_style_session(
        session_path, n_poses=n_session, noise_t=he_noise_t, noise_r=he_noise_r
    )
    stages.append(
        {
            "name": "handeye_session_export",
            "returncode": 0,
            "elapsed_s": 0.0,
            "artifact": str(session_path),
            "summary": {
                "consistency_rmse_opencv": session["consistency_rmse_opencv"],
                "consistency_rmse_refined": session["consistency_rmse_refined"],
                "err_t_opencv_m": session["err_t_opencv_m"],
                "err_t_refined_m": session["err_t_refined_m"],
                "n": len(session["samples"]),
            },
        }
    )

    # 2–4) Offline evals
    stages.append(
        {
            "name": "handeye_ba",
            **run_module(
                "eval_handeye_ba.py",
                ["--trials", str(he_trials), "--noise-t", str(he_noise_t), "--noise-r", str(he_noise_r)],
                py_path,
            ),
        }
    )
    stages.append(
        {
            "name": "handeye_ceres_style",
            **run_module(
                "eval_handeye_ceres_style.py",
                ["--trials", str(he_trials), "--noise-t", str(he_noise_t), "--noise-r", str(he_noise_r)],
                py_path,
            ),
        }
    )
    stages.append(
        {
            "name": "vision_grasp_apparent",
            **run_module("eval_vision_grasp.py", ["--trials", str(vision_trials)], py_path),
        }
    )
    stages.append(
        {
            "name": "vision_grasp_fixed_z",
            **run_module(
                "eval_vision_grasp.py",
                ["--trials", str(vision_trials), "--fixed-z"],
                py_path,
            ),
        }
    )

    # Stress sweep (occlusion / noise cliffs) — always for full, light for quick
    if args.profile == "full":
        stages.append(
            {
                "name": "stress_sweep",
                **run_module(
                    "eval_stress.py",
                    ["--vision-trials", "24", "--handeye-trials", "10"],
                    py_path,
                ),
            }
        )
        stages.append(
            {
                "name": "handeye_bakeoff",
                **run_module("eval_handeye_bakeoff.py", ["--trials", "20"], py_path),
            }
        )
        stages.append(
            {
                "name": "grasp_uncertainty",
                **run_module(
                    "eval_grasp_uncertainty.py",
                    ["--trials", "30", "--occlusion", "0.75"],
                    py_path,
                ),
            }
        )
    else:
        stages.append(
            {
                "name": "stress_sweep",
                **run_module(
                    "eval_stress.py",
                    ["--vision-trials", "12", "--handeye-trials", "6"],
                    py_path,
                ),
            }
        )

    if args.with_planning:
        stages.append(
            {
                "name": "planning_obstacle",
                **run_module(
                    "eval_planning_obstacle.py",
                    ["--trials", str(plan_trials)],
                    py_path,
                ),
            }
        )
        stages.append(
            {
                "name": "planning_pareto",
                **run_module(
                    "eval_planning_pareto.py",
                    ["--quick", "--trials", "3"],
                    py_path,
                ),
            }
        )

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report_md = out / "SHIFT_REPORT.md"
    report_json = out / "shift.json"
    write_report(report_md, meta, stages)
    report_json.write_text(
        json.dumps({"meta": meta, "stages": stages}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(json.dumps({"shift_id": shift_id, "dir": str(out), "report": str(report_md)}, indent=2))
    failed = [s for s in stages if s.get("returncode", 0) != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
