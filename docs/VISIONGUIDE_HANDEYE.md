# Hand-eye residuals: industrial Ceres stack → this package

Reference industrial stack (`visionguide`-style):

| Module | Role |
|--------|------|
| Eye-to-hand Ceres problem | PnP reprojection + multi-pose consistency |
| Eye-in-hand Ceres problem | Same family for camera-on-wrist |
| Halcon / OpenCV hand-eye | Closed-form or nonlinear init |
| Capture UI | Images + robot poses for calibration |

**Ceres** = Google nonlinear least squares. Typical costs:
1. Reprojection of board points through the hand-eye chain
2. Multi-pose consistency (object pose agrees across robot configurations)

## Mapping in this package

| Module | Role |
|--------|------|
| `handeye_ba.py` | SE3 pose-chain residual LM |
| `handeye_ceres_style.py` | Pair-consistency residual (eye-to-hand) |
| `eval_handeye_ceres_style.py` | OpenCV vs consistency-LM under noise |

```bash
ros2 run arm_system eval_handeye_ceres_style.py --trials 15
```

On real hardware, feed `ToolInBase` + `objInCam` samples into the same residual API.
