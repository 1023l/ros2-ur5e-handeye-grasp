#!/usr/bin/env python3
"""Collect geometric-teacher demos for AGV roll-transfer ACT training.

Offline — no MoveIt required. Produces episode.json + images under --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    # allow running without sourced install
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from arm_system.policy.geometry_teacher import collect_demos

    p = argparse.ArgumentParser(description="Collect AGV roll-transfer demos")
    p.add_argument(
        "--out",
        type=Path,
        default=root / "results" / "agv_demos",
        help="Demo dataset root",
    )
    p.add_argument("--n", type=int, default=80, help="Successful episodes to collect")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps-per-seg", type=int, default=3)
    p.add_argument(
        "--dock-heavy-frac",
        type=float,
        default=0.45,
        help="Fraction of demos sampled with agv_dock_perturb DR",
    )
    args = p.parse_args()

    summary = collect_demos(
        args.out,
        n=args.n,
        seed0=args.seed,
        steps_per_seg=args.steps_per_seg,
        dock_heavy_frac=args.dock_heavy_frac,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["success"] < max(1, args.n // 4):
        print("WARN: too few successes — check scene/teacher thresholds", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
