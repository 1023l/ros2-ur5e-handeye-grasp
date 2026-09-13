# 约束规划权衡（Pareto）

在障碍膨胀（间隙代理）× 规划时限 × 姿态约束松紧上扫网格，得到成功率 vs 轨迹时长的非支配前沿，再选工作点——不要只报单一 `plan_rate`。

## 怎么跑

```bash
ros2 launch arm_system ur5e_moveit_demo.launch.py launch_rviz:=false
ros2 run arm_system eval_planning_pareto.py --quick --trials 4
# 更密网格：
# ros2 run arm_system eval_planning_pareto.py --trials 6
```

输出：`results/planning_pareto_summary.json`

| 字段 | 含义 |
|------|------|
| `clearance_inflate_m` | 障碍盒膨胀 → 自由空间变紧（间隙代理） |
| `planning_time_s` | OMPL 允许规划时间 |
| `orientation_tol_rad` | 工具姿态约束松紧 |
| `plan_rate` | 该格点成功率 |
| `mean_traj_duration_s` | 成功轨迹平均时长 |
| `pareto_front` | 非支配解（高成功率、短轨迹） |
| `recommended` | 前沿上优先推荐的工作点 |

## 和旧评测关系

- `eval_planning_obstacle.py`：有/无障碍二值对比
- `eval_planning_pareto.py`：约束权衡曲线
