"""Geometric teacher for AGV roll transfer (waypoint + EE-delta actions).

Mirrors industrial vision-guide style: believe a pick XY / bay XY, execute
fixed phase waypoints. Used as:
  1) demo generator (successful transfers under mild noise)
  2) residual prior for ACT (network learns correction deltas)
  3) eval baseline under dock / perception perturbations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from arm_system.policy.constants import (
    ACTION_DIM,
    AGV_BAY_XY,
    APPROACH_Z,
    GRASP_Z,
    PLACE_Z,
    STATE_DIM,
)
from arm_system.policy.scene_agv import AgvSceneGT, in_agv_bay, make_agv_scene, perceive_roll
from arm_system.policy.schema import Episode, EpisodeMeta, StepRecord, new_episode_id, save_image


def _aa_from_tool_down() -> np.ndarray:
    # tool pointing down ≈ rot x by pi → axis-angle (pi, 0, 0)
    return np.array([np.pi, 0.0, 0.0], dtype=np.float64)


@dataclass
class EEState:
    xyz: np.ndarray  # (3,)
    aa: np.ndarray  # (3,)
    gripper: float  # 0 open, 1 closed

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.xyz, self.aa, [self.gripper]])


def _interp_xyz(a: np.ndarray, b: np.ndarray, n: int) -> List[np.ndarray]:
    if n <= 1:
        return [b.copy()]
    out = []
    for i in range(1, n + 1):
        t = i / n
        out.append(a * (1 - t) + b * t)
    return out


def build_waypoint_path(
    pick_xy: Tuple[float, float],
    place_xy: Tuple[float, float],
    *,
    start_xyz: Optional[np.ndarray] = None,
    steps_per_seg: int = 4,
) -> List[Tuple[str, EEState]]:
    """Phase-tagged EE waypoints (same structure as pick_place_core)."""
    sx = start_xyz if start_xyz is not None else np.array([0.3, 0.0, 0.35])
    px, py = pick_xy
    qx, qy = place_xy
    aa = _aa_from_tool_down()
    path: List[Tuple[str, EEState]] = []

    segments = [
        ("approach", np.array([px, py, APPROACH_Z]), 0.0),
        ("grasp", np.array([px, py, GRASP_Z]), 0.0),
        ("grasp_close", np.array([px, py, GRASP_Z]), 1.0),
        ("lift", np.array([px, py, APPROACH_Z]), 1.0),
        ("transit", np.array([qx, qy, APPROACH_Z]), 1.0),
        ("place", np.array([qx, qy, PLACE_Z]), 1.0),
        ("place_open", np.array([qx, qy, PLACE_Z]), 0.0),
        ("retreat", np.array([qx, qy, APPROACH_Z]), 0.0),
        ("home", np.array([0.3, 0.0, 0.35]), 0.0),
    ]

    cur = EEState(xyz=sx.copy(), aa=aa.copy(), gripper=0.0)
    for phase, target, grip in segments:
        for xyz in _interp_xyz(cur.xyz, target, steps_per_seg):
            g = cur.gripper
            path.append((phase, EEState(xyz=xyz.copy(), aa=aa.copy(), gripper=g)))
            cur = path[-1][1]
        cur = EEState(xyz=cur.xyz.copy(), aa=aa.copy(), gripper=float(grip))
        path.append((phase, cur))
    return path


def build_place_only_path(
    place_xy: Tuple[float, float],
    *,
    start: EEState,
    steps_per_seg: int = 3,
) -> List[Tuple[str, EEState]]:
    """From post-lift pose to AGV bay (no re-grasp)."""
    qx, qy = place_xy
    aa = _aa_from_tool_down()
    path: List[Tuple[str, EEState]] = [("hold", start)]
    segments = [
        ("transit", np.array([qx, qy, APPROACH_Z]), 1.0),
        ("place", np.array([qx, qy, PLACE_Z]), 1.0),
        ("place_open", np.array([qx, qy, PLACE_Z]), 0.0),
        ("retreat", np.array([qx, qy, APPROACH_Z]), 0.0),
        ("home", np.array([0.3, 0.0, 0.35]), 0.0),
    ]
    cur = start
    for phase, target, grip in segments:
        for xyz in _interp_xyz(cur.xyz, target, steps_per_seg):
            path.append((phase, EEState(xyz=xyz.copy(), aa=aa.copy(), gripper=cur.gripper)))
            cur = path[-1][1]
        cur = EEState(xyz=cur.xyz.copy(), aa=aa.copy(), gripper=float(grip))
        path.append((phase, cur))
    return path


def ee_delta_action(prev: EEState, nxt: EEState) -> np.ndarray:
    d = np.zeros(ACTION_DIM, dtype=np.float64)
    d[:3] = nxt.xyz - prev.xyz
    d[3:6] = nxt.aa - prev.aa
    d[6] = nxt.gripper
    return d


def state_vector(
    ee: EEState,
    roll_xy_est: Tuple[float, float],
    agv_xy_est: Tuple[float, float],
) -> np.ndarray:
    s = np.zeros(STATE_DIM, dtype=np.float64)
    s[:3] = ee.xyz
    s[3:6] = ee.aa
    s[6] = ee.gripper
    s[7:9] = roll_xy_est
    s[9:11] = agv_xy_est
    return s


def render_with_ee_marker(scene_img: np.ndarray, ee_uv: Optional[Tuple[int, int]]) -> np.ndarray:
    img = scene_img.copy()
    if ee_uv is not None:
        cv2 = __import__("cv2")
        cv2.circle(img, ee_uv, 7, (0, 255, 255), 2)
        cv2.drawMarker(img, ee_uv, (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
    return img


def _ee_to_uv(xyz: np.ndarray) -> Tuple[int, int]:
    from arm_system.policy.scene_agv import _world_to_uv

    u, v = _world_to_uv(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return int(u), int(v)


@dataclass
class RolloutResult:
    success: bool
    failure_mode: str
    final_place_xy: Tuple[float, float]
    n_steps: int
    episode: Optional[Episode] = None


def classify_failure(
    *,
    grasped: bool,
    placed_in_bay: bool,
    hit_table_early: bool = False,
    occluded: bool = False,
) -> str:
    if hit_table_early:
        return "collision_proxy"
    if not grasped:
        return "miss_grasp" if not occluded else "occlusion_miss"
    if not placed_in_bay:
        return "miss_agv_bay"
    return ""


def run_geometric_episode(
    *,
    seed: int,
    profile_perturb,
    belief_noise: Optional[float] = None,
    record: bool = False,
    out_root=None,
    steps_per_seg: int = 3,
    exec_noise: float = 0.0,
    expert_for_demo: bool = False,
) -> RolloutResult:
    """Execute geometric open-loop against frozen scene GT.

    Modes:
      - eval baseline (expert_for_demo=False): follow noisy belief waypoints
      - demo collect (expert_for_demo=True): follow near-oracle expert path to
        true roll/bay, while storing geom_prior from noisy belief — residual
        ACT then learns vision-conditioned corrections, not identity.
    """
    rng = np.random.default_rng(seed)
    gt = make_agv_scene(seed=seed, perturb=profile_perturb)
    pnoise = (
        belief_noise
        if belief_noise is not None
        else float(gt.extras.get("perception_xy_noise", 0.0))
    )
    roll_belief = perceive_roll(gt, rng=rng, perception_xy_noise=pnoise)
    # Planned dock = map nominal (+ tiny read noise). True bay may have shifted
    # after AGV late-dock — geometric open-loop cannot see that; ACT can.
    dock_read_noise = float(getattr(profile_perturb, "agv_dock_xy_noise", 0.0)) * 0.15
    bay_belief = (
        AGV_BAY_XY[0] + float(rng.normal(0, max(0.002, dock_read_noise))),
        AGV_BAY_XY[1] + float(rng.normal(0, max(0.002, dock_read_noise))),
    )

    belief_path = build_waypoint_path(roll_belief, bay_belief, steps_per_seg=steps_per_seg)
    if expert_for_demo:
        # slight jitter on GT so demos are not a single frozen trajectory
        roll_ex = (
            gt.roll_xyz[0] + float(rng.normal(0, 0.0015)),
            gt.roll_xyz[1] + float(rng.normal(0, 0.0015)),
        )
        bay_ex = (
            gt.agv_bay_xy[0] + float(rng.normal(0, 0.0015)),
            gt.agv_bay_xy[1] + float(rng.normal(0, 0.0015)),
        )
        path = build_waypoint_path(roll_ex, bay_ex, steps_per_seg=steps_per_seg)
    else:
        path = belief_path

    # align prior actions to belief schedule (same length by construction)
    prior_actions = [
        ee_delta_action(belief_path[i][1], belief_path[i + 1][1])
        for i in range(len(belief_path) - 1)
    ]

    grasped = False
    place_xy = (float(path[0][1].xyz[0]), float(path[0][1].xyz[1]))
    prev = path[0][1]
    steps: List[StepRecord] = []
    ep_id = new_episode_id()
    ep_dir = None
    if record and out_root is not None:
        from pathlib import Path

        ep_dir = Path(out_root) / ep_id
        ep_dir.mkdir(parents=True, exist_ok=True)

    for t, (phase, nxt) in enumerate(path[1:], start=0):
        if exec_noise > 0:
            jitter = rng.normal(0, exec_noise, size=3)
            nxt = EEState(
                xyz=nxt.xyz + jitter,
                aa=nxt.aa.copy(),
                gripper=nxt.gripper,
            )
        action = ee_delta_action(prev, nxt)
        prior = prior_actions[min(t, len(prior_actions) - 1)]
        st = state_vector(prev, roll_belief, bay_belief)
        img = render_with_ee_marker(gt.image_bgr, _ee_to_uv(prev.xyz))
        if record and ep_dir is not None:
            rel = save_image(ep_dir, t, img)
        else:
            rel = f"images/{t:04d}.png"
        steps.append(
            StepRecord(
                t=t,
                image=rel,
                state=st.tolist(),
                action=action.tolist(),
                ee_xyz=prev.xyz.tolist(),
                gripper=float(prev.gripper),
                phase=phase,
                geom_prior=prior.tolist(),
            )
        )
        if phase.startswith("grasp") and nxt.gripper >= 0.5:
            dist = np.linalg.norm(nxt.xyz[:2] - np.array(gt.roll_xyz[:2]))
            grasped = dist < 0.045
        if phase.startswith("place"):
            place_xy = (float(nxt.xyz[0]), float(nxt.xyz[1]))
        prev = nxt

    placed = bool(grasped and in_agv_bay(place_xy, gt.agv_bay_xy))
    if placed and prev.gripper > 0.5:
        placed = False
    fail = classify_failure(grasped=bool(grasped), placed_in_bay=placed, occluded=bool(gt.occluded))
    success = bool(placed and fail == "")

    ep = None
    if record:
        meta = EpisodeMeta(
            episode_id=ep_id,
            success=success,
            failure_mode=fail,
            profile=getattr(profile_perturb, "name", "custom"),
            seed=seed,
            n_steps=len(steps),
            roll_xyz=list(gt.roll_xyz),
            agv_bay_xy=list(gt.agv_bay_xy),
            dock_offset_xy=[
                float(gt.agv_bay_xy[0] - bay_belief[0]),
                float(gt.agv_bay_xy[1] - bay_belief[1]),
            ],
            perturb={
                "perception_xy_noise": pnoise,
                "agv_dock_xy_noise": float(getattr(profile_perturb, "agv_dock_xy_noise", 0)),
            },
            notes="expert_demo_residual" if expert_for_demo else "geometric_teacher",
            extra={"expert_for_demo": expert_for_demo, "bay_belief": list(bay_belief)},
        )
        ep = Episode(meta=meta, steps=steps)
        if ep_dir is not None:
            ep.save(ep_dir.parent)

    return RolloutResult(
        success=success,
        failure_mode=fail,
        final_place_xy=place_xy,
        n_steps=len(steps),
        episode=ep,
    )


def collect_demos(
    out_root,
    *,
    n: int = 80,
    seed0: int = 0,
    steps_per_seg: int = 3,
    mild_perturb=None,
    dock_heavy_frac: float = 0.45,
) -> dict:
    """Collect successful geometric demos under mild domain randomization."""
    from pathlib import Path

    from arm_system.policy.schema import DatasetIndex
    from arm_system.policy.scene_agv import EVAL_PROFILES, ScenePerturb

    out_root = Path(out_root)
    rng = np.random.default_rng(seed0 + 99)
    out_root.mkdir(parents=True, exist_ok=True)
    index = DatasetIndex(out_root)
    # Scene DR for demos; belief noise is intentional so residual targets ≠ 0
    mild = mild_perturb or ScenePerturb(
        roll_xy_noise=0.006,
        agv_dock_xy_noise=0.04,  # expert→true bay; prior→nominal map
        perception_xy_noise=0.018,
        occlusion=0.12,
        clutter=3,
        lighting_gain=0.95,
    )
    ok = 0
    fail_modes: dict = {}
    attempted = 0
    seed = seed0
    heavy = EVAL_PROFILES["agv_dock_perturb"]
    while ok < n and attempted < n * 8:
        attempted += 1
        use_heavy = float(rng.random()) < float(dock_heavy_frac)
        perturb = heavy if use_heavy else mild
        tag = "collect_dock_heavy" if use_heavy else "collect_mild"
        r = run_geometric_episode(
            seed=seed,
            profile_perturb=perturb,
            belief_noise=0.018,
            record=True,
            out_root=out_root,
            steps_per_seg=steps_per_seg,
            exec_noise=0.0008,
            expert_for_demo=True,
        )
        seed += 1
        if r.episode is None:
            continue
        r.episode.meta.profile = tag
        r.episode.save(out_root)
        if r.success:
            ok += 1
            index.append(r.episode.meta)
        else:
            fail_modes[r.failure_mode] = fail_modes.get(r.failure_mode, 0) + 1
            # still index failures for analysis but train uses success_only
            index.append(r.episode.meta)
    return {
        "requested": n,
        "success": ok,
        "attempted": attempted,
        "fail_modes": fail_modes,
        "out_root": str(out_root),
    }
