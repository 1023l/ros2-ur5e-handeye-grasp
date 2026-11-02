# WSL 环境（arm_system / MoveIt）

## 用哪个 Ubuntu？

| 发行版 | 说明 |
|--------|------|
| **Ubuntu-22.04** | ✅ **用这个**。已装 **ROS2 Humble + MoveIt**，与本包 README 一致 |
| Ubuntu（默认） | 24.04，无 Humble 时勿用来跑 MoveIt |
| docker-desktop | 容器用，不是日常 ROS 工作区 |

项目路径（Windows 与 WSL 通用）：

```text
/mnt/c/Users/Administrator/Documents/trae_projects/arm_system
```

## Windows 换行（CRLF）导致 `python3\r` 报错

项目在 `/mnt/c/...` 时，若 launch 报 `/usr/bin/env: 'python3\r': No such file or directory`，在 **Ubuntu-22.04** 项目根执行：

```bash
cd /mnt/c/Users/Administrator/Documents/trae_projects/arm_system
find arm_system scripts -name '*.py' -exec sed -i 's/\r$//' {} +
```

然后重开 launch / `ros2 run`。仓库已加 `.gitattributes` 尽量保持 `*.py` 为 LF。

## 一次性编译安装

```bash
wsl -d Ubuntu-22.04
source /opt/ros/humble/setup.bash
cd /mnt/c/Users/Administrator/Documents/trae_projects/arm_system
# 若 arm_system 在 colcon 工作空间里：
# cd ~/ros2_ws && colcon build --packages-select arm_system --symlink-install
# 或在本包上级 workspace 编译

# 本仓库也可直接用 Python 脚本（不需 colcon）：
export PYTHONPATH=/mnt/c/Users/Administrator/Documents/trae_projects/arm_system:$PYTHONPATH
python3 scripts/collect_agv_demos.py --n 80
```

脚本 `scripts/wsl_build_arm_system.sh` 可在 22.04 里自动 source Humble 并检查依赖。

## WSL 里 RViz 没有窗口？

WSL2 要用 **WSLg** 才能把 RViz 画到 Windows 桌面。若 `launch_rviz:=true` 但看不到机械臂：

1. 确认是 **Windows 11**（或已启用 WSLg 的 Win10），且存在目录 `/mnt/wslg`。
2. **不要**从纯 SSH/无 GUI 会话进 WSL；用 Windows 自带的 **「Ubuntu-22.04」** 终端或 Cursor 集成终端即可。
3. 改代码后重建：`cd ~/ros2_ws && colcon build --packages-select arm_system --symlink-install`，再重启 launch。  
   `ur5e_moveit_demo.launch.py` 会在检测到 WSLg 时自动设置 `DISPLAY=:0`、`WAYLAND_DISPLAY=wayland-0`。
4. 仍无界面时，MoveIt 已跑着的情况下 **另开终端** 手动起 RViz（必须先 `source`，且脚本需已安装）：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
ros2 run arm_system start_rviz.sh
# 若报 No such file ... start_rviz.sh → 在工作区重建：
# cd ~/ros2_ws && colcon build --packages-select arm_system --symlink-install
# 若仍 127/ENOENT → CRLF：在项目根 find scripts -name '*.sh' -exec sed -i 's/\r$//' {} +
```

也可不经过 ros2 run，直接：

```bash
bash ~/ros2_ws/src/arm_system/scripts/start_rviz.sh
```

5. 自检 GUI：`sudo apt install -y x11-apps && xeyes`（应弹出眼睛窗口；没有则是系统/WSLg 问题，不是 ROS）。

录 demo 时终端 1 用 `launch_rviz:=true`；只跑 `pick_place` 可继续 `launch_rviz:=false` 省资源。

## MoveIt + 策略停靠 + 抓放（在线）

**第 0 步 — 清场（`pick_place` 报 `more than one action server` 或 error -4 时必做）**

MoveIt **只能起一份**。重复 launch 会出现 **2 个 `/move_action`**，`pick_place` 必挂。

```bash
bash ~/ros2_ws/src/arm_system/scripts/wsl_demo_reset.sh
# 或：关掉所有 ROS 终端后重开
source /opt/ros/humble/setup.bash
ros2 action info /move_action   # Action servers 必须是 1
```

**终端 1 — MoveIt（不要开 RViz，避免和手动 RViz 重复）**

```bash
wsl -d Ubuntu-22.04
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch arm_system ur5e_moveit_demo.launch.py launch_rviz:=false
```

**终端 1b — RViz 只看机械臂（可选）**

```bash
export DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
bash ~/ros2_ws/src/arm_system/scripts/start_rviz.sh
```

跑 `pick_place` 时 **不要** 在 RViz 里 Plan/Execute 或拖交互球。

**终端 2 — 感知 + 手眼 + 假夹爪 + ACT 停靠节点（一条 launch）**

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch arm_system arm_agv_online.launch.py
```

可选改 ckpt：`ckpt_dir:=/path/to/act_ckpt`

**终端 3 — 一次抓放（自动调 `/arm/policy/correct_place` 取 `place_xy`）**

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run arm_system pick_place.py
```

固定放料点、不走策略时：

```bash
ros2 run arm_system pick_place.py --ros-args -r __node:=policy_place_client -p use_policy:=false
```

**调试 service（可选）**

```bash
ros2 service call /arm/policy/correct_place std_srvs/srv/Trigger {}
```

改代码后在工作区：`cd ~/ros2_ws && colcon build --packages-select arm_system --symlink-install`

**POC 阶段：** 离线 `eval_act_vs_baseline.py` + `sweep_dock_stress.py` 锁数字；MoveIt 用于演示「同协议在线执行」。

## Windows 上训练 PyTorch

CPU/CUDA 训练可在 Windows 直接跑脚本；ROS 只在 WSL 22.04。ckpt 路径两边共享 `results/act_ckpt`。
