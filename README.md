# arm_system — UR5e eye-to-hand grasp simulation (ROS2 Humble)

MoveIt2 fake hardware + hard-scene vision + hand-eye BA / Ceres-style residuals + grasp ranking + planning metrics + **residual ACT for AGV dock placement**.

Simulation practice loop: [`docs/SIM_AS_REAL.md`](docs/SIM_AS_REAL.md). WSL setup: [`docs/WSL_ROS2.md`](docs/WSL_ROS2.md).

## One-shot shift

```bash
# Offline (no MoveIt required)
ros2 run arm_system run_sim_shift.py --profile quick
# Full + planning (start MoveIt fake first):
# ros2 run arm_system run_sim_shift.py --profile full --with-planning

ros2 run arm_system eval_stress.py
ros2 run arm_system eval_handeye_bakeoff.py --trials 30
ros2 run arm_system eval_grasp_uncertainty.py --trials 40 --occlusion 0.75
```

Reports under `results/shifts/shift_*/` (local runs) and checked-in eval summaries under `results/act_eval/`, `results/ablations/`, `results/stress_curve/`.

## Evals (split)

### 1) Hand-eye

Aligned with industrial Ceres-style residuals — see [`docs/VISIONGUIDE_HANDEYE.md`](docs/VISIONGUIDE_HANDEYE.md).

```bash
ros2 run arm_system eval_handeye_ba.py --trials 20 --noise-t 0.004 --noise-r 0.02
ros2 run arm_system eval_handeye_ceres_style.py --trials 15
ros2 run arm_system calibrate_handeye.py --mode synthetic --noise-t 0.004 --noise-r 0.02
```

### 2) Vision grasp

```bash
ros2 run arm_system eval_vision_grasp.py --trials 50
ros2 run arm_system eval_vision_grasp.py --trials 50 --fixed-z
```

### 3) Planning (needs MoveIt)

```bash
ros2 launch arm_system ur5e_moveit_demo.launch.py launch_rviz:=false
ros2 run arm_system eval_planning_obstacle.py --trials 12
ros2 run arm_system eval_planning_pareto.py --quick --trials 4
```

See [`docs/PLANNING_PARETO.md`](docs/PLANNING_PARETO.md).

## Residual ACT — roll transfer to AGV dock

Pick with geometry/vision waypoints; **late dock placement** fused from bin vision + ACT dock head (not end-to-end black box). Details: [`docs/AGV_ROLL_TRANSFER_ACT.md`](docs/AGV_ROLL_TRANSFER_ACT.md).

```bash
python scripts/collect_agv_demos.py --n 80 --out results/agv_demos
python scripts/train_act.py --data results/agv_demos --out results/act_ckpt --epochs 40
python scripts/eval_act_vs_baseline.py --ckpt results/act_ckpt --out results/act_eval --trials 80
python scripts/run_ablations.py
```

Pinned offline numbers (80 trials/profile): see `results/act_eval/summary.json` — e.g. `agv_dock_perturb` geometric **31.2%** vs ACT fusion **98.8%**.

**Online (WSL + MoveIt fake):**

```bash
ros2 launch arm_system arm_agv_online.launch.py
ros2 run arm_system pick_place.py
```

Policy service `/arm/policy/correct_place` → full pick → place → home.

## Demo

```bash
ros2 launch arm_system ur5e_moveit_demo.launch.py
ros2 launch arm_system arm_vision_demo.launch.py
ros2 run arm_system arm_agent.py
ros2 run arm_system say.py "抓放"
```
