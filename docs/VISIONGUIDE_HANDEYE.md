# Mapping: visionguide (KUKA 视觉引导) → arm_system

Your `visionguide` project already has the industrial Ceres hand-eye stack:

| File | Role |
|------|------|
| `produce/CeresEyeToHandProblem.h` | 眼在手外：PnP 重投影 + `OptBaseInCamCeres` 多位姿一致性 |
| `produce/CeresEyeInHandProblem.h` | 眼在手上：PnP 重投影 + `OptCameraCeres` |
| `Calibration/DRCalibration.cpp` | Halcon `CalibrateHandEye` 初值（nonlinear） |
| `DialogHandEye.cpp` | UI：采图 / 读机器人位姿 / 调标定 |

**Ceres 是什么：** Google 的非线性最小二乘库。你在手眼里用它优化的不是“随便拟合”，而是：
1. **重投影残差**：标定板 3D 点经手眼链投到图像，和检测 uv 差；
2. **多位姿一致性**：不同机器人位姿下，物体在工具/基座系应一致，位姿差当残差。

这和 KUKA 强相关：`DialogHandEye` + 机器人位姿文件就是产线标定流程；Ceres 是其中的优化器。

## arm_system 里的对应

| arm_system | 对应 visionguide 思想 |
|------------|----------------------|
| `handeye_ba.py` | SE3 位姿链残差 LM（仿真底板） |
| `handeye_ceres_style.py` | **直接对齐** `calHandEyePose` / pair consistency |
| `eval_handeye_ceres_style.py` | 噪声下对比 OpenCV vs 一致性 LM |

```bash
ros2 run arm_system eval_handeye_ceres_style.py --trials 15
```

真机时：把 visionguide 导出的 `ToolInBase` + `objInCam` 样本灌进同一 residual API，指标口径就和产线一致。
