# arm_system — 机械臂主线（深度向）

本地只提交本包。底板 + 难场景指标 + BA 手眼 + 多候选抓取 + 绕障规划评测。

**没有真机也按「仿真产线班次」攒具身经验** → 见 [`docs/SIM_AS_REAL.md`](docs/SIM_AS_REAL.md)。

> 真机缺的是力觉/标定漂移/节拍；仿真等价练的是：**决策链 + 工业口径指标 + 失败归因**。面试官认闭环，不认“有没有摸过铁”。

## 一键班次（优先）

```bash
# 离线：标定会话导出 + 手眼/视觉消融报告（不需 MoveIt）
ros2 run arm_system run_sim_shift.py --profile quick
# 或 full；MoveIt fake 起来后可加：
# ros2 run arm_system run_sim_shift.py --profile full --with-planning

# 加压扫描（遮挡↑ / 噪声↑ 找悬崖，再针对性改）
ros2 run arm_system eval_stress.py

# 资深口径：方法 bake-off + 不确定性抓取消融
ros2 run arm_system eval_handeye_bakeoff.py --trials 30
ros2 run arm_system eval_grasp_uncertainty.py --trials 40 --occlusion 0.75
```

级别判断见 [`docs/LEVELING.md`](docs/LEVELING.md)。报告在 `arm_system/results/shifts/shift_*/SHIFT_REPORT.md`；压力结果 `results/stress_sweep_summary.json`。

## 深度评测（拆开跑）

### 1) 手眼：OpenCV 初值 + 残差精炼（对齐 visionguide / Ceres）

你以前的工业手眼在 `visionguide`（Halcon 初值 + **Google Ceres** 重投影/多位姿一致性），不是空喊的「Ceres」。详见 `docs/VISIONGUIDE_HANDEYE.md`。

```bash
# SE3 位姿链 BA（仿真底板）
ros2 run arm_system eval_handeye_ba.py --trials 20 --noise-t 0.004 --noise-r 0.02

# 对齐 visionguide OptBaseInCam 的 pair-consistency 残差
ros2 run arm_system eval_handeye_ceres_style.py --trials 15

# 标定写 YAML（默认开 BA）
ros2 run arm_system calibrate_handeye.py --mode synthetic --noise-t 0.004 --noise-r 0.02
```

### 2) 视觉抓取：难场景 + 表观尺寸估深 + 多候选评分

```bash
ros2 run arm_system eval_vision_grasp.py --trials 50
ros2 run arm_system eval_vision_grasp.py --trials 50 --fixed-z
```

在线默认 `scene_mode:=hard`，debug 图会标 `#1/#2/#3` 候选。

### 3) 规划绕障 + 约束权衡曲线（需 MoveIt）

```bash
ros2 launch arm_system ur5e_moveit_demo.launch.py launch_rviz:=false
ros2 run arm_system eval_planning_obstacle.py --trials 12
# 高级：间隙代理 × 规划时限 × 姿态约束 → Pareto 前沿
ros2 run arm_system eval_planning_pareto.py --quick --trials 4
```

详见 [`docs/PLANNING_PARETO.md`](docs/PLANNING_PARETO.md)。对比无障碍 / 有障碍的 `plan_rate_*`；Pareto 看 `recommended` 工作点。

结果都在 `arm_system/results/`（git 忽略）。

## 演示底板

```bash
ros2 launch arm_system ur5e_moveit_demo.launch.py
ros2 launch arm_system arm_vision_demo.launch.py
ros2 run arm_system arm_agent.py
ros2 run arm_system say.py "抓放"
```

## 路线

1. ~~流程底板~~  
2. ~~难场景 / 估深消融~~  
3. ~~BA 手眼残差 / 多候选 / 绕障规划评测~~  
4. ~~无真机「仿真产线班次」闭环（`run_sim_shift`）~~  
5. ~~加压扫描 + 检测/深度/手眼门控优化~~  
6. ~~约束规划权衡曲线（Pareto）~~  
7. 再往后：真机样本灌入同一 session schema、更强抓取网络、可复用分包  
