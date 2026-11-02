"""AGV roll-transfer policy learning (ACT + geometric baseline).

This is the learning layer on top of the geometric pick-place stack:
collect teacher demos → train ACT → eval under dock/pose perturbations.
"""

from arm_system.policy.constants import ACTION_DIM, CHUNK_SIZE, STATE_DIM

__all__ = ["ACTION_DIM", "STATE_DIM", "CHUNK_SIZE"]
