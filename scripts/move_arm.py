#!/usr/bin/env python3
"""CLI: move UR5e via MoveIt2.

Examples (demo launch must already be running):

  ros2 run arm_system move_arm.py --named up
  ros2 run arm_system move_arm.py --named home
  ros2 run arm_system move_arm.py --joints 0 -1.57 0 -1.0 0 0
  ros2 run arm_system move_arm.py --pose 0.3 0.0 0.4 3.14 0 0
  ros2 run arm_system move_arm.py --named up --plan-only
"""

from __future__ import annotations

import argparse
import sys

import rclpy

from arm_system.moveit_client import ARM_JOINTS, MoveItArmClient, list_named


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Move UR5e with MoveIt2")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--named",
        choices=list_named(),
        help="SRDF named joint target (home / up)",
    )
    g.add_argument(
        "--joints",
        nargs=6,
        type=float,
        metavar=("PAN", "LIFT", "ELBOW", "W1", "W2", "W3"),
        help="Six arm joint positions [rad]",
    )
    g.add_argument(
        "--pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="tool0 pose in base_link: xyz [m] + rpy [rad]",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan but do not execute",
    )
    p.add_argument("--vel", type=float, default=0.3, help="Velocity scale (0-1)")
    p.add_argument("--acc", type=float, default=0.3, help="Acceleration scale (0-1)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rclpy.init()
    node = MoveItArmClient()
    try:
        if not node.wait_ready():
            return 2

        execute = not args.plan_only
        if args.named:
            code = node.move_named(
                args.named,
                execute=execute,
                velocity_scale=args.vel,
                accel_scale=args.acc,
            )
        elif args.joints is not None:
            target = dict(zip(ARM_JOINTS, args.joints))
            code = node.move_joints(
                target,
                execute=execute,
                velocity_scale=args.vel,
                accel_scale=args.acc,
            )
        else:
            x, y, z, roll, pitch, yaw = args.pose
            code = node.move_pose_rpy(
                x,
                y,
                z,
                roll,
                pitch,
                yaw,
                execute=execute,
                velocity_scale=args.vel,
                accel_scale=args.acc,
            )

        return 0 if code == 1 else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
