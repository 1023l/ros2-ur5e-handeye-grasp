#!/usr/bin/env python3
"""Planning success with vs without collision obstacles (needs MoveIt running).

  ros2 launch arm_system ur5e_moveit_demo.launch.py launch_rviz:=false
  ros2 run arm_system eval_planning_obstacle.py --trials 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node

from arm_system.metrics import MetricsLogger, TrialRecord
from arm_system.moveit_client import SUCCESS, MoveItArmClient
from arm_system.planning_scene_obstacles import add_box_obstacle, clear_obstacle
from arm_system.pose_utils import tool_down_pose


def results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def plan_to(arm: MoveItArmClient, pose: Pose) -> bool:
    code = arm.move_pose(
        pose,
        execute=False,  # plan only
        velocity_scale=0.2,
        accel_scale=0.2,
        planning_time=5.0,
    )
    return code == SUCCESS


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10)
    args = p.parse_args(argv)

    # Targets that often interact with a box near (0.35, 0.12)
    goals = [
        (0.35, 0.00, 0.22),
        (0.32, 0.18, 0.22),
        (0.40, 0.10, 0.22),
        (0.28, 0.08, 0.20),
        (0.38, -0.05, 0.22),
        (0.33, 0.15, 0.18),
    ]

    rclpy.init()
    helper = Node("arm_plan_eval_helper")
    arm = MoveItArmClient(node_name="arm_plan_eval_moveit")
    logger = MetricsLogger(results_dir(), run_name="planning_obstacle")

    try:
        if not arm.wait_ready(15.0):
            print("move_action unavailable — start ur5e_moveit_demo first")
            return 2

        clear_obstacle(helper)
        # Baseline: no obstacle
        for i in range(args.trials):
            x, y, z = goals[i % len(goals)]
            ok = plan_to(arm, tool_down_pose(x, y, z))
            logger.log(
                TrialRecord(
                    trial_id=i,
                    timestamp=time.time(),
                    scene_mode="no_obstacle",
                    target_label="pose",
                    detected=True,
                    plan_success=ok,
                    failure_mode="" if ok else "plan_fail",
                    extra={"xyz": [x, y, z]},
                )
            )

        # With obstacle
        add_box_obstacle(helper, xyz=(0.35, 0.12, 0.08), size=(0.08, 0.08, 0.16))
        time.sleep(0.5)
        for i in range(args.trials):
            x, y, z = goals[i % len(goals)]
            ok = plan_to(arm, tool_down_pose(x, y, z))
            logger.log(
                TrialRecord(
                    trial_id=100 + i,
                    timestamp=time.time(),
                    scene_mode="with_obstacle",
                    target_label="pose",
                    detected=True,
                    plan_success=ok,
                    failure_mode="" if ok else "plan_fail",
                    extra={"xyz": [x, y, z]},
                )
            )
        clear_obstacle(helper)

        # Split summary
        base = [r for r in logger.records if r.scene_mode == "no_obstacle"]
        hard = [r for r in logger.records if r.scene_mode == "with_obstacle"]
        summary = {
            "n_each": args.trials,
            "plan_rate_no_obstacle": sum(1 for r in base if r.plan_success) / max(len(base), 1),
            "plan_rate_with_obstacle": sum(1 for r in hard if r.plan_success) / max(len(hard), 1),
            "jsonl": str(logger.jsonl_path),
        }
        logger.summarize()
        logger.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        helper.destroy_node()
        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
