#!/usr/bin/env python3
"""Sweep AGV dock noise → stress curve (geometric vs ACT modes)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from arm_system.policy.stress_eval import sweep_dock_stress

    p = argparse.ArgumentParser(description="Dock noise stress sweep")
    p.add_argument("--ckpt", type=Path, default=root / "results" / "act_ckpt")
    p.add_argument("--out", type=Path, default=root / "results" / "stress_curve")
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--seed0", type=int, default=5000)
    p.add_argument(
        "--noise",
        nargs="*",
        type=float,
        default=None,
        help="e.g. 0 0.02 0.04 0.06 0.08",
    )
    args = p.parse_args()

    if not (args.ckpt / "best.pt").exists():
        print(f"missing ckpt: {args.ckpt / 'best.pt'}", file=sys.stderr)
        return 2

    summary = sweep_dock_stress(
        args.ckpt,
        args.out,
        noise_levels=args.noise,
        trials=args.trials,
        seed0=args.seed0,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nReport: {args.out / 'STRESS_CURVE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
