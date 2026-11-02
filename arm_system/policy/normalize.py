"""Action / state normalization stats (z-score) persisted with checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from arm_system.policy.constants import ACTION_DIM, STATE_DIM


@dataclass
class NormStats:
    action_mean: List[float]
    action_std: List[float]
    state_mean: List[float]
    state_std: List[float]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "NormStats":
        return NormStats(**json.loads(Path(path).read_text(encoding="utf-8")))

    def norm_action(self, a: np.ndarray) -> np.ndarray:
        return (a - np.asarray(self.action_mean)) / (np.asarray(self.action_std) + 1e-6)

    def denorm_action(self, a: np.ndarray) -> np.ndarray:
        return a * (np.asarray(self.action_std) + 1e-6) + np.asarray(self.action_mean)

    def norm_state(self, s: np.ndarray) -> np.ndarray:
        return (s - np.asarray(self.state_mean)) / (np.asarray(self.state_std) + 1e-6)


def compute_stats(actions: Iterable[np.ndarray], states: Iterable[np.ndarray]) -> NormStats:
    A = np.stack(list(actions), axis=0)
    S = np.stack(list(states), axis=0)
    assert A.shape[-1] == ACTION_DIM and S.shape[-1] == STATE_DIM
    return NormStats(
        action_mean=A.mean(0).tolist(),
        action_std=np.maximum(A.std(0), 1e-3).tolist(),
        state_mean=S.mean(0).tolist(),
        state_std=np.maximum(S.std(0), 1e-3).tolist(),
    )
