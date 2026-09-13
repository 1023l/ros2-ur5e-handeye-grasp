# arm_system — UR5e eye-to-hand grasp simulation (ROS2 Humble)

MoveIt2 fake hardware + hard-scene vision + hand-eye BA / Ceres-style residuals + grasp ranking + planning metrics.

Simulation practice loop: [`docs/SIM_AS_REAL.md`](docs/SIM_AS_REAL.md).

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

Reports under `results/shifts/shift_*/` and `results/*_summary.json`.

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

## Demo

```bash
ros2 launch arm_system ur5e_moveit_demo.launch.py
ros2 launch arm_system arm_vision_demo.launch.py
ros2 run arm_system arm_agent.py
ros2 run arm_system say.py "抓放"
```
