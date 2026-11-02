"""Shared constants for AGV roll-transfer ACT stack."""

from __future__ import annotations

# EE delta (dx,dy,dz, dax,day,daz) + gripper cmd in [0,1] (0=open, 1=closed)
ACTION_DIM = 7
# proprio: ee xyz + axis-angle (3) + gripper + roll_xy_est + agv_xy_est
STATE_DIM = 6 + 1 + 2 + 2  # 11
CHUNK_SIZE = 16

# Workspace (base_link meters) — eye-to-hand table + AGV bay
TABLE_Z = 0.10
APPROACH_Z = 0.24
GRASP_Z = 0.11
PLACE_Z = 0.12

# Nominal AGV bay center in base frame (docked vehicle load slot)
AGV_BAY_XY = (0.42, -0.18)
# Tight slot: late-dock offset must be corrected (not forgiven by a huge bay)
AGV_BAY_HALF = (0.04, 0.05)

# Image
IMG_H = 120
IMG_W = 160
IMG_FULL_H = 480
IMG_FULL_W = 640

SCHEMA_VERSION = 2
TASK_ID = "agv_roll_transfer"
