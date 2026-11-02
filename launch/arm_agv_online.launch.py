"""Vision stack + ACT policy dock node (MoveIt 仍单独起 ur5e_moveit_demo)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("arm_system")
    ckpt_dir = LaunchConfiguration("ckpt_dir")
    dock_mode = LaunchConfiguration("dock_mode")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ckpt_dir",
                default_value=(
                    "/mnt/c/Users/Administrator/Documents/trae_projects/"
                    "arm_system/results/act_ckpt"
                ),
            ),
            DeclareLaunchArgument("dock_mode", default_value="fusion"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([pkg, "/launch/arm_vision_demo.launch.py"]),
            ),
            Node(
                package="arm_system",
                executable="policy_dock_node.py",
                name="policy_dock_node",
                output="screen",
                parameters=[
                    {"ckpt_dir": ckpt_dir, "dock_mode": dock_mode},
                ],
            ),
        ]
    )
