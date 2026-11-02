# AGV Roll Transfer — ACT vs Geometric

- ckpt: `C:\Users\Administrator\Documents\trae_projects\arm_system\results\act_ckpt`
- trials/profile: 80

| Profile | Geometric | ACT (primary) | Δ (pp) |
|---------|-----------|-----------------|--------|
| nominal | 100.0% | 100.0% | +0.0 |
| agv_dock_perturb | 31.2% | 98.8% | +67.5 |
| clutter_occlusion | 91.2% | 91.2% | +0.0 |

## Failure modes (ACT primary)
### nominal
- (none)

### agv_dock_perturb
- `miss_grasp`: 1

### clutter_occlusion
- `occlusion_miss`: 4
- `miss_grasp`: 3

Success = grasp within 5cm of true roll center AND release inside true AGV bay.
