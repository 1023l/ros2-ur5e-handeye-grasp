"""Temporal ensembling for ACT action chunks (paper inference trick).

Each step the policy predicts a chunk of future actions; we keep a FIFO of
predictions and exponentially weight earlier predictions for the current step.
Stabilizes jitter without a second network.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


class TemporalEnsembler:
    def __init__(self, chunk_size: int, exp_weight: float = 0.5) -> None:
        self.chunk_size = chunk_size
        self.exp_weight = exp_weight
        self._buf: Deque[np.ndarray] = deque(maxlen=chunk_size)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, chunk: np.ndarray) -> np.ndarray:
        """
        chunk: (T, action_dim) predicted from current observation.
        returns ensembled action for the immediate next step (action_dim,)
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        self._buf.appendleft(chunk)
        # gather predictions that claim to apply at "now"
        votes = []
        weights = []
        for age, pred in enumerate(self._buf):
            if age >= pred.shape[0]:
                continue
            votes.append(pred[age])
            weights.append(np.exp(-self.exp_weight * age))
        if not votes:
            return chunk[0]
        w = np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-8)
        return np.sum([wi * vi for wi, vi in zip(w, votes)], axis=0)
