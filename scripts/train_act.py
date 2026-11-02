#!/usr/bin/env python3
"""Train residual ACT on AGV roll-transfer demos (single GPU/CPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from arm_system.policy.train_loop import train_act

    p = argparse.ArgumentParser(description="Train ACT for AGV roll transfer")
    p.add_argument("--data", type=Path, default=root / "results" / "agv_demos")
    p.add_argument("--out", type=Path, default=root / "results" / "act_ckpt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no-residual", action="store_true", help="Disable residual-over-geom prior")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--dock-weight", type=float, default=2.0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if not args.data.exists():
        print(f"missing demos: {args.data} — run collect_agv_demos.py first", file=sys.stderr)
        return 2

    summary = train_act(
        args.data,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        residual=not args.no_residual,
        device=args.device,
        train_augment=not args.no_augment,
        dock_weight=args.dock_weight,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
