# 没有真机，怎么攒「类似真机」的具身经验

真机缺的是：**真实接触力、相机标定漂移、产线节拍、安全联锁**。  
仿真能等价练的是：**同一条决策链 + 同一套指标口径 + 可复现失败分析**。面试官看的是你有没有这套闭环，不是有没有拍过工业现场照。

## 你这条仿真产线对应真机什么

| 真机产线 | 你现在的 sim |
|----------|--------------|
| 眼在手外采多姿态标定板 | `handeye_session.json` 合成多位姿 + 噪声 |
| Halcon/OpenCV 初值 + Ceres 精炼 | `eval_handeye_*` / `handeye_ceres_style` |
| 视觉找料、遮挡、估深 | `hard_scene` + `eval_vision_grasp` |
| 多抓取候选择优 | `grasp_ranking` |
| 有料箱/护栏绕障 | MoveIt + `eval_planning_obstacle` |
| 班次报表 / 不良归因 | `run_sim_shift.py` → `SHIFT_REPORT.md` |
| 自然语言下任务 | `arm_agent` / `say.py` |

工业手眼残差语言对齐你以前的 `visionguide`（见 `VISIONGUIDE_HANDEYE.md`）。

## 每周怎么练（建议节奏）

1. **离线班次**  
   `ros2 run arm_system run_sim_shift.py --profile full`
2. **加压找弱项（不是瞎搞坏）**  
   `ros2 run arm_system eval_stress.py`  
   把遮挡/标定噪声逐步加大，看检测率、位姿合格率、手眼误差在哪一级垮掉，再改那一块。
3. **在线班次**  
   起 MoveIt fake → `--with-planning`；偶尔 `say.py "抓放"`。
4. **口头能讲清**：一条残差、一个悬崖点、一次消融结论。

## 简历一句话（可改）

> ROS2 + MoveIt2 搭建眼在手外仿真抓取产线：工业口径手眼精炼、难场景视觉与抓取评分、绕障规划评测；用可复现班次报告做指标闭环（无真机阶段）。

有真机后再把 `handeye_session.json` 换成实采样本即可，链路不用推倒重来。
