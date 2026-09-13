#!/usr/bin/env python3
"""Constrained planning Pareto sweep: clearance × planning_time × orient_tol.

Senior-shaped deliverable: not one plan_rate, but a tradeoff table +
non-dominated front (maximize plan_rate, minimize traj duration).

Clearance is proxied by inflating the collision box (larger obstacle ⇒
less free space / tighter practical clearance).

  ros2 launch arm_system ur5e_moveit_demo.launch.py launch_rviz:=false
  ros2 run arm_system eval_planning_pareto.py --trials 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node

from arm_system.moveit_client import MoveItArmClient
from arm_system.planning_scene_obstacles import add_box_obstacle, clear_obstacle
from arm_system.pose_utils import tool_down_pose


def results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


GOALS = [
    (0.35, 0.00, 0.22),
    (0.32, 0.18, 0.22),
    (0.40, 0.10, 0.22),
    (0.28, 0.08, 0.20),
    (0.38, -0.05, 0.22),
    (0.33, 0.15, 0.18),
]

BASE_XYZ = (0.35, 0.12, 0.08)
BASE_SIZE = (0.08, 0.08, 0.16)


def pareto_front(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Non-dominated: higher plan_rate better, lower mean_traj_duration better."""
    front: List[Dict[str, Any]] = []
    for a in rows:
        dominated = False
        for b in rows:
            if a is b:
                continue
            better_or_eq_rate = b["plan_rate"] >= a["plan_rate"]
            better_or_eq_dur = b["mean_traj_duration_s"] <= a["mean_traj_duration_s"]
            strictly = (
                b["plan_rate"] > a["plan_rate"]
                or b["mean_traj_duration_s"] < a["mean_traj_duration_s"]
            )
            if better_or_eq_rate and better_or_eq_dur and strictly:
                dominated = True
                break
        if not dominated:
            front.append(a)
    front.sort(key=lambda r: (-r["plan_rate"], r["mean_traj_duration_s"]))
    return front


def run_cell(
    arm: MoveItArmClient,
    helper: Node,
    *,
    inflate_m: float,
    planning_time: float,
    orient_tol: float,
    trials: int,
) -> Dict[str, Any]:
    size = (
        BASE_SIZE[0] + 2.0 * inflate_m,
        BASE_SIZE[1] + 2.0 * inflate_m,
        BASE_SIZE[2] + 2.0 * inflate_m,
    )
    clear_obstacle(helper)
    add_box_obstacle(helper, xyz=BASE_XYZ, size=size)
    time.sleep(0.35)

    oks = 0
    walls, durs, paths = [], [], []
    for i in range(trials):
        x, y, z = GOALS[i % len(GOALS)]
        m = arm.plan_pose_metrics(
            tool_down_pose(x, y, z),
            execute=False,
            velocity_scale=0.25,
            accel_scale=0.25,
            planning_time=planning_time,
            orientation_tol=orient_tol,
            num_attempts=8,
        )
        if m.success:
            oks += 1
            walls.append(m.wall_time_s)
            durs.append(m.traj_duration_s)
            paths.append(m.path_length_rad)

    clear_obstacle(helper)
    n = max(trials, 1)
    return {
        "clearance_inflate_m": inflate_m,
        "obstacle_size": list(size),
        "planning_time_s": planning_time,
        "orientation_tol_rad": orient_tol,
        "trials": trials,
        "plan_rate": oks / n,
        "mean_wall_s": sum(walls) / len(walls) if walls else None,
        "mean_traj_duration_s": sum(durs) / len(durs) if durs else 1e9,
        "mean_path_length_rad": sum(paths) / len(paths) if paths else None,
        "n_success": oks,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Planning constraint Pareto sweep")
    p.add_argument("--trials", type=int, default=4, help="Goals per grid cell")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller grid (faster)",
    )
    args = p.parse_args(argv)

    if args.quick:
        inflates = [0.0, 0.03, 0.06]
        plan_times = [2.0, 8.0]
        orients = [0.1, 0.25]
    else:
        inflates = [0.0, 0.02, 0.04, 0.06]
        plan_times = [2.0, 5.0, 10.0]
        orients = [0.08, 0.20]

    rclpy.init()
    helper = Node("arm_plan_pareto_helper")
    arm = MoveItArmClient(node_name="arm_plan_pareto_moveit")
    try:
        if not arm.wait_ready(15.0):
            print(
                json.dumps(
                    {
                        "error": "move_action unavailable",
                        "hint": "ros2 launch arm_system ur5e_moveit_demo.launch.py launch_rviz:=false",
                    },
                    indent=2,
                )
            )
            return 2

        # Consistent start state
        arm.move_named("home", execute=True, velocity_scale=0.3, accel_scale=0.3)
        time.sleep(0.5)

        rows: List[Dict[str, Any]] = []
        for inf in inflates:
            for pt in plan_times:
                for ot in orients:
                    print(
                        f"cell inflate={inf:.3f} plan_t={pt} orient={ot} ...",
                        flush=True,
                    )
                    rows.append(
                        run_cell(
                            arm,
                            helper,
                            inflate_m=inf,
                            planning_time=pt,
                            orient_tol=ot,
                            trials=args.trials,
                        )
                    )

        front = pareto_front(rows)
        # Prefer high rate then short traj among front for "recommended"
        recommended = front[0] if front else None
        out = {
            "grid": rows,
            "pareto_front": front,
            "recommended": recommended,
            "axes": {
                "clearance_inflate_m": "obstacle inflation (proxy: tighter free space)",
                "planning_time_s": "OMPL allowed planning time",
                "orientation_tol_rad": "tool orientation constraint tightness",
            },
            "note": (
                "Report a Pareto front over clearance proxy × planning budget × "
                "orientation constraint; pick an operating point — not a single plan_rate."
            ),
        }
        path = results_dir() / "planning_pareto_summary.json"
        results_dir().mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        print(f"wrote {path}")
        return 0
    finally:
        clear_obstacle(helper)
        helper.destroy_node()
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
