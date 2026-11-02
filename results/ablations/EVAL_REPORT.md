# AGV Roll Transfer — ACT vs Geometric

- ckpt: `C:\Users\Administrator\Documents\trae_projects\arm_system\results\act_ckpt`
- trials/profile: 40

| Profile | Geometric | ACT (primary) | Δ (pp) |
|---------|-----------|-----------------|--------|
| agv_dock_perturb | 40.0% | 40.0% | +0.0 |
| nominal | 100.0% | 100.0% | +0.0 |

## Failure modes (ACT primary)
### agv_dock_perturb
- `miss_agv_bay`: 24

### nominal
- (none)

Success = grasp within 5cm of true roll center AND release inside true AGV bay.
