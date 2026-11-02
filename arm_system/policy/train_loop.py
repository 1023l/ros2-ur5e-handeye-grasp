"""ACT training loop with KL anneal, grad clip, checkpointing."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

from arm_system.policy.act_model import ACTPolicy, count_params, kl_anneal
from arm_system.policy.constants import CHUNK_SIZE
from arm_system.policy.dataset import AgvChunkDataset


def train_act(
    data_root: Path,
    out_dir: Path,
    *,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-4,
    residual: bool = True,
    device: Optional[str] = None,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    seed: int = 0,
    train_augment: bool = True,
    dock_weight: float = 2.0,
) -> Dict[str, Any]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds_full = AgvChunkDataset(
        data_root, chunk_size=CHUNK_SIZE, residual=residual, train_augment=False
    )
    stats = ds_full.stats
    n_val = max(1, int(len(ds_full) * val_ratio))
    n_train = len(ds_full) - n_val
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds_full), generator=gen).tolist()
    train_i = perm[:n_train]
    val_i = perm[n_train:]
    ds_train = AgvChunkDataset(
        data_root,
        chunk_size=CHUNK_SIZE,
        residual=residual,
        train_augment=train_augment,
        stats=stats,
        seed=seed + 1,
    )
    ds_val = AgvChunkDataset(
        data_root,
        chunk_size=CHUNK_SIZE,
        residual=residual,
        train_augment=False,
        stats=stats,
    )
    train_ds = Subset(ds_train, train_i)
    val_ds = Subset(ds_val, val_i)
    ds = ds_full
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = ACTPolicy(residual=residual).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    ds.stats.save(out_dir / "norm_stats.json")
    history = []
    global_step = 0
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_l1 = tr_kl = 0.0
        n_tr = 0
        for batch in train_loader:
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)
            prior = batch["geom_prior"].to(device) if residual else None
            dock = batch["dock_offset"].to(device)
            kw = kl_anneal(global_step, warmup=1500)
            out = model(image, state, action_chunk=actions, geom_prior=prior, kl_weight=kw)
            dock_l1 = torch.nn.functional.l1_loss(out["dock_offset"], dock)
            loss = out["loss"] + dock_weight * dock_l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            global_step += 1
            bs = image.shape[0]
            tr_loss += float(loss.item()) * bs
            tr_l1 += float(out["l1"].item()) * bs
            tr_kl += float(out["kl"].item()) * bs
            n_tr += bs

        model.eval()
        va_loss = va_l1 = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                state = batch["state"].to(device)
                actions = batch["actions"].to(device)
                prior = batch["geom_prior"].to(device) if residual else None
                dock = batch["dock_offset"].to(device)
                out = model(image, state, action_chunk=actions, geom_prior=prior, kl_weight=1.0)
                dock_l1 = torch.nn.functional.l1_loss(out["dock_offset"], dock)
                bs = image.shape[0]
                va_loss += float((out["loss"] + dock_weight * dock_l1).item()) * bs
                va_l1 += float(out["l1"].item()) * bs
                n_va += bs
        sched.step()

        row = {
            "epoch": epoch,
            "train_loss": tr_loss / max(1, n_tr),
            "train_l1": tr_l1 / max(1, n_tr),
            "train_kl": tr_kl / max(1, n_tr),
            "val_loss": va_loss / max(1, n_va),
            "val_l1": va_l1 / max(1, n_va),
            "lr": float(sched.get_last_lr()[0]),
            "n_params": count_params(model),
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] train_l1={row['train_l1']:.4f} "
            f"val_l1={row['val_l1']:.4f} kl={row['train_kl']:.4f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "residual": residual,
            "epoch": epoch,
            "val_l1": row["val_l1"],
        }
        torch.save(ckpt, out_dir / "last.pt")
        if row["val_l1"] < best_val:
            best_val = row["val_l1"]
            torch.save(ckpt, out_dir / "best.pt")

    summary = {
        "best_val_l1": best_val,
        "epochs": epochs,
        "residual": residual,
        "train_augment": train_augment,
        "dock_weight": dock_weight,
        "n_train": n_train,
        "n_val": n_val,
        "n_params": count_params(model),
        "device": device,
        "finished_at": time.time(),
    }
    (out_dir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def load_policy(ckpt_dir: Path, device: Optional[str] = None) -> tuple:
    ckpt_dir = Path(ckpt_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from arm_system.policy.normalize import NormStats

    stats = NormStats.load(ckpt_dir / "norm_stats.json")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    residual = bool(ckpt.get("residual", True))
    model = ACTPolicy(residual=residual).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, stats, residual, device
