#!/usr/bin/env python3
"""Ablation eval: fusion / vision_only / dock_only / nominal vs geometric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from arm_system.policy.eval_protocol import evaluate_compare
    from arm_system.policy.scene_agv import EVAL_PROFILES

    p = argparse.ArgumentParser(description="ACT controller ablations")
    p.add_argument("--ckpt", type=Path, default=root / "results" / "act_ckpt")
    p.add_argument("--out", type=Path, default=root / "results" / "ablations")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--seed0", type=int, default=3000)
    p.add_argument(
        "--profiles",
        nargs="*",
        default=["agv_dock_perturb", "nominal"],
    )
    p.add_argument(
        "--modes",
        nargs="*",
        default=["nominal", "vision_only", "dock_only", "fusion"],
    )
    args = p.parse_args()

    if not (args.ckpt / "best.pt").exists():
        print(f"missing ckpt: {args.ckpt / 'best.pt'}", file=sys.stderr)
        return 2

    for m in args.profiles:
        if m not in EVAL_PROFILES:
            print(f"unknown profile: {m}", file=sys.stderr)
            return 2

    summary = evaluate_compare(
        args.ckpt,
        args.out,
        profiles=args.profiles,
        trials=args.trials,
        seed0=args.seed0,
        ablation_modes=args.modes,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReport: {args.out / 'EVAL_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
