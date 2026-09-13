# Simulation as a stand-in for line work

Real hardware still differs in contact force, calibration drift, cycle time, and safety interlocks.
This package focuses on what simulation *can* match: the same decision chain, the same metrics, and reproducible failure analysis.

## Pipeline mapping

| Industrial step | This repo |
|-----------------|-----------|
| Multi-pose eye-to-hand board capture | synthetic `handeye_session.json` + noise |
| OpenCV / Halcon init + nonlinear refine | `eval_handeye_*`, `handeye_ceres_style` |
| Find object under clutter / occlusion | `hard_scene` + `eval_vision_grasp` |
| Rank grasp hypotheses | `grasp_ranking` / `grasp_uncertainty` |
| Plan around fixtures | MoveIt + `eval_planning_obstacle` / `eval_planning_pareto` |
| Shift report | `run_sim_shift.py` → `results/shifts/*/SHIFT_REPORT.md` |
| Simple language command | `arm_agent` / `say.py` |

Hand-eye residual language aligns with industrial Ceres-style consistency (see `VISIONGUIDE_HANDEYE.md`).

## Suggested practice loop

1. Offline shift: `ros2 run arm_system run_sim_shift.py --profile full`
2. Stress sweep: `ros2 run arm_system eval_stress.py`
3. With MoveIt fake: `--with-planning` and occasional `say.py` commands
4. Track one residual, one cliff point, one ablation result per iteration
