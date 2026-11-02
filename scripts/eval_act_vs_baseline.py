#!/usr/bin/env python3
"""Compare ACT vs geometric baseline under AGV dock / clutter profiles."""

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

    p = argparse.ArgumentParser(description="ACT vs geometric eval")
    p.add_argument("--ckpt", type=Path, default=root / "results" / "act_ckpt")
    p.add_argument("--out", type=Path, default=root / "results" / "act_eval")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument(
        "--profiles",
        nargs="*",
        default=list(EVAL_PROFILES.keys()),
        help=f"subset of {list(EVAL_PROFILES.keys())}",
    )
    p.add_argument("--no-ensemble", action="store_true")
    p.add_argument(
        "--dock-mode",
        type=str,
        default="fusion",
        choices=["fusion", "vision_only", "dock_only", "nominal"],
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="Run all dock modes vs geometric",
    )
    args = p.parse_args()

    if not (args.ckpt / "best.pt").exists():
        print(f"missing checkpoint: {args.ckpt / 'best.pt'}", file=sys.stderr)
        return 2

    ablation_modes = None
    if args.ablation:
        ablation_modes = ["nominal", "vision_only", "dock_only", "fusion"]
    summary = evaluate_compare(
        args.ckpt,
        args.out,
        profiles=args.profiles,
        trials=args.trials,
        seed0=args.seed0,
        use_ensemble=not args.no_ensemble,
        dock_mode=args.dock_mode,
        ablation_modes=ablation_modes,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReport: {args.out / 'EVAL_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
