"""Chunk dataset over recorded AGV-transfer episodes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from arm_system.policy.augment import AugmentConfig, augment_rgb
from arm_system.policy.constants import ACTION_DIM, CHUNK_SIZE, IMG_H, IMG_W, STATE_DIM
from arm_system.policy.normalize import NormStats, compute_stats
from arm_system.policy.schema import DatasetIndex, Episode


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


class AgvChunkDataset(Dataset):
    def __init__(
        self,
        root: Path,
        *,
        chunk_size: int = CHUNK_SIZE,
        success_only: bool = True,
        stats: Optional[NormStats] = None,
        residual: bool = True,
        train_augment: bool = False,
        augment: Optional[AugmentConfig] = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.residual = residual
        self.train_augment = train_augment
        self.augment = augment or AugmentConfig()
        self._rng = np.random.default_rng(seed)
        self.episodes: List[Episode] = []
        for ep_dir in DatasetIndex(self.root).list_episodes(success_only=success_only):
            if (ep_dir / "episode.json").exists():
                self.episodes.append(Episode.load(ep_dir))
        self.samples: List[Tuple[int, int]] = []  # (ep_idx, t)
        for ei, ep in enumerate(self.episodes):
            n = len(ep.steps)
            for t in range(max(0, n - 1)):
                self.samples.append((ei, t))
        if not self.samples:
            raise RuntimeError(f"No samples under {self.root}")

        if stats is None:
            actions, states = [], []
            for ei, t in self.samples:
                actions.append(np.asarray(self.episodes[ei].steps[t].action, dtype=np.float64))
                states.append(np.asarray(self.episodes[ei].steps[t].state, dtype=np.float64))
            self.stats = compute_stats(actions, states)
        else:
            self.stats = stats

    def __len__(self) -> int:
        return len(self.samples)

    def _chunk(self, ep: Episode, t: int, key: str) -> np.ndarray:
        out = np.zeros((self.chunk_size, ACTION_DIM), dtype=np.float32)
        for k in range(self.chunk_size):
            j = min(t + k, len(ep.steps) - 1)
            step = ep.steps[j]
            vec = getattr(step, key)
            if vec is None:
                vec = step.action
            out[k] = np.asarray(vec, dtype=np.float32)
        return out

    def __getitem__(self, idx: int):
        ei, t = self.samples[idx]
        ep = self.episodes[ei]
        step = ep.steps[t]
        assert ep.root is not None
        img = _load_rgb(ep.root / step.image)
        if self.train_augment:
            img = augment_rgb(img, self._rng, self.augment)
        state = np.asarray(step.state, dtype=np.float32)
        actions = self._chunk(ep, t, "action")
        prior = self._chunk(ep, t, "geom_prior") if self.residual else np.zeros_like(actions)

        # normalize
        state_n = self.stats.norm_state(state).astype(np.float32)
        actions_n = np.stack([self.stats.norm_action(a) for a in actions]).astype(np.float32)
        prior_n = np.stack([self.stats.norm_action(a) for a in prior]).astype(np.float32)

        dock = np.asarray(ep.meta.dock_offset_xy, dtype=np.float32)
        if dock.shape != (2,):
            dock = np.zeros(2, dtype=np.float32)
        return {
            "image": torch.from_numpy(img).permute(2, 0, 1),
            "state": torch.from_numpy(state_n),
            "actions": torch.from_numpy(actions_n),
            "geom_prior": torch.from_numpy(prior_n),
            "dock_offset": torch.from_numpy(dock),
        }
