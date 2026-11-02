#!/usr/bin/env python3
"""Small hyperparam grid for ACT (build training intuition)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from arm_system.policy.stress_eval import sweep_dock_stress
    from arm_system.policy.train_loop import train_act

    p = argparse.ArgumentParser(description="Train grid + quick dock stress score")
    p.add_argument("--data", type=Path, default=root / "results" / "agv_demos")
    p.add_argument("--out-root", type=Path, default=root / "results" / "train_sweep")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--stress-trials", type=int, default=15)
    args = p.parse_args()

    if not args.data.exists():
        print("run collect_agv_demos first", file=sys.stderr)
        return 2

    grid = [
        {"lr": 1e-4, "dock_weight": 1.0, "train_augment": False},
        {"lr": 1e-4, "dock_weight": 2.0, "train_augment": True},
        {"lr": 3e-4, "dock_weight": 2.0, "train_augment": True},
        {"lr": 1e-4, "dock_weight": 4.0, "train_augment": True},
    ]
    rows = []
    for i, cfg in enumerate(grid):
        run_id = f"run_{i:02d}_lr{cfg['lr']}_dw{cfg['dock_weight']}_aug{int(cfg['train_augment'])}"
        ckpt_dir = args.out_root / run_id
        t0 = time.time()
        train_summary = train_act(
            args.data,
            ckpt_dir,
            epochs=args.epochs,
            batch_size=64,
            lr=cfg["lr"],
            dock_weight=cfg["dock_weight"],
            train_augment=cfg["train_augment"],
        )
        stress = sweep_dock_stress(
            ckpt_dir,
            ckpt_dir / "stress_quick",
            noise_levels=[0.025, 0.055],
            trials=args.stress_trials,
            seed0=8000 + i * 100,
            controller_modes=["geometric", "fusion"],
        )
        # score: fusion@0.055 - geometric@0.055
        r55 = next(x for x in stress["rows"] if abs(x["agv_dock_xy_noise"] - 0.055) < 1e-6)
        score = r55["fusion"] - r55["geometric"]
        rows.append(
            {
                "run_id": run_id,
                "cfg": cfg,
                "best_val_l1": train_summary.get("best_val_l1"),
                "fusion@0.055": r55["fusion"],
                "geom@0.055": r55["geometric"],
                "delta@0.055": score,
                "wall_s": time.time() - t0,
            }
        )
        print(json.dumps(rows[-1], indent=2))

    out = {"runs": rows, "created_at": time.time()}
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "sweep_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    best = max(rows, key=lambda r: r["delta@0.055"])
    print(f"\nBest by delta@0.055: {best['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
