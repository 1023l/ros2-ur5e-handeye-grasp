#!/usr/bin/env python3
"""Insert/clear a demo collision box in MoveIt planning scene."""

from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node

from arm_system.planning_scene_obstacles import add_box_obstacle, clear_obstacle


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true")
    p.add_argument("--x", type=float, default=0.35)
    p.add_argument("--y", type=float, default=0.12)
    p.add_argument("--z", type=float, default=0.08)
    args = p.parse_args(argv)
    rclpy.init()
    node = Node("arm_scene_obstacle_cli")
    try:
        if args.clear:
            ok = clear_obstacle(node)
        else:
            ok = add_box_obstacle(node, xyz=(args.x, args.y, args.z))
        return 0 if ok else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
