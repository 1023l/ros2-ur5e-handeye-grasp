# 卷装上 AGV：策略学习（ACT）闭环

业务切口（无真机仿真 POC）：

> 检测完成后的布卷 → 机械臂拣放到 **已停稳的 AGV 载具位**。  
> AGV 行驶与到站倾倒不在本包范围内。

这不是「调个 API 跑 demo」：自研 **ACT + CVAE + 时序集成 + 残差几何先验**，并与几何开环基线做同协议对比评测。

## 架构

| 模块 | 路径 | 作用 |
|------|------|------|
| 场景 | `policy/scene_agv.py` | 布卷 + AGV 料斗合成图；三档扰动；料斗经典视觉估计 |
| 几何教师 | `policy/geometry_teacher.py` | 工业 waypoint 示教；残差 prior = 地图标称停靠 |
| Schema | `policy/schema.py` | episode.json + images + manifest + dock_offset |
| ACT | `policy/act_model.py` | CNN→Transformer；CVAE chunk；**dock Δxy 头**；残差动作 |
| 训练 | `policy/train_loop.py` | KL anneal + dock L1；grad clip；cosine LR |
| 推理 | `policy/eval_protocol.py` | 几何抓取 + 视觉/ACT 融合停靠修正 + 重规划放料 |
| 评测 | 同上 | geometric vs ACT；bootstrap CI；失败模式 |

**控制器（工业残差，不是瞎端到端）：**  
抓取段跟几何 waypoint；抬升后用「料斗检测 ⊕ ACT dock 头」修正 AGV 目标，再几何放到修正位。

## 一键流程（离线，不需 MoveIt）

```bash
# 1) 自采示教（专家轨迹 → 真值；prior → 噪声信念）
python scripts/collect_agv_demos.py --n 80 --out results/agv_demos

# 2) 训练残差 ACT
python scripts/train_act.py --data results/agv_demos --out results/act_ckpt --epochs 40

# 3) 对比评测（nominal / agv_dock_perturb / clutter_occlusion）
python scripts/eval_act_vs_baseline.py --ckpt results/act_ckpt --out results/act_eval --trials 40
```

报告：`results/act_eval/EVAL_REPORT.md`。

## 成功判据（写进简历前必须一致）

1. 夹爪闭合时 EE 距真值布卷中心 **≤ 5 cm**  
2. 释放时 EE 落在真值 AGV bay 矩形内  

几何基线：仅用噪声感知信念做开环 waypoint（对应「视觉定姿 + 固定路径」）。  
ACT：图像条件 + 残差修正，在 **AGV 停靠偏移 / 感知噪声** 下应高于基线。

## 评测档

| Profile | 含义 |
|---------|------|
| `nominal` | 轻扰动，基线应较高 |
| `agv_dock_perturb` | 载具停靠晚偏移 —— ACT 该拉开差距 |
| `clutter_occlusion` | 杂乱 + 遮挡 —— 写失败分析 |

## 本地 smoke 结果（可复现，写简历前请重跑锁定）

`python scripts/eval_act_vs_baseline.py --trials 40` 一次样例：

| Profile | Geometric | ACT（几何抓取 + 料斗视觉/ACT 停靠修正） | Δ |
|---------|-----------|----------------------------------------|---|
| nominal | 100% | 100% | 0 |
| agv_dock_perturb | 40% | 100% | **+60 pp** |
| clutter_occlusion | 90% | 90% | 0 |

主叙事：地图标称停靠开环在 AGV 晚停靠时塌；视觉条件修正后回到满成功率。

## 简历口径（有数字再写）

> 卷装上 AGV 转序：自采 N 条示教；残差策略在停靠扰动下成功率 xx%，相对几何开环 +yy pp；失败模式见 EVAL_REPORT。

以你机器上最新 `results/act_eval/summary.json` 为准，不要抄文档里的样例表。

## 80～90 分：实验与在线（必读）

- 实验台账：[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)
- WSL / MoveIt：[`WSL_ROS2.md`](WSL_ROS2.md)

```bash
python scripts/sweep_dock_stress.py --trials 30
python scripts/run_ablations.py --trials 40
python scripts/sweep_train.py   # 小网格，耗时较长
```
