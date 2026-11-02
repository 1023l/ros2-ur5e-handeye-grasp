"""AGV dock correction modes (classical vision / learned / fusion)."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

DockMode = Literal["fusion", "vision_only", "dock_only", "nominal"]

VISION_CONF_THRESH = 0.25


def fuse_dock_bay(
    bay_belief: Tuple[float, float],
    *,
    vis_xy: Tuple[float, float],
    vis_conf: float,
    learn_xy: Tuple[float, float],
    mode: DockMode = "fusion",
) -> Tuple[Tuple[float, float], float, float]:
    """
    Returns (corrected_bay, dx, dy) relative to bay_belief.
    """
    bx, by = bay_belief
    if mode == "nominal":
        return (bx, by), 0.0, 0.0

    if mode == "vision_only":
        if vis_conf >= VISION_CONF_THRESH:
            corrected = (float(vis_xy[0]), float(vis_xy[1]))
        else:
            corrected = (bx, by)
        return corrected, corrected[0] - bx, corrected[1] - by

    if mode == "dock_only":
        corrected = (float(learn_xy[0]), float(learn_xy[1]))
        dx = learn_xy[0] - bx
        dy = learn_xy[1] - by
        if (dx * dx + dy * dy) ** 0.5 < 0.02:
            return (bx, by), 0.0, 0.0
        return corrected, dx, dy

    # fusion
    if vis_conf >= VISION_CONF_THRESH:
        w = float(np.clip(vis_conf, VISION_CONF_THRESH, 0.9))
        corrected = (
            (1 - w) * learn_xy[0] + w * vis_xy[0],
            (1 - w) * learn_xy[1] + w * vis_xy[1],
        )
    else:
        dx = learn_xy[0] - bx
        dy = learn_xy[1] - by
        if (dx * dx + dy * dy) ** 0.5 < 0.02:
            corrected = (bx, by)
        else:
            corrected = (float(learn_xy[0]), float(learn_xy[1]))
    return corrected, corrected[0] - bx, corrected[1] - by
