"""
UR5e + MoveIt2 最小可动演示（假硬件，无需真机）。

用法：
  ros2 launch arm_system ur5e_moveit_demo.launch.py

RViz 操作：
  1) 顶部工具栏点「Interact」（交互箭头图标）
  2) 末端附近会出现可拖拽目标球/环
  3) 拖到新位姿 → Plan → Execute
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = LaunchConfiguration('ur_type')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    launch_rviz = LaunchConfiguration('launch_rviz')
    robot_ip = LaunchConfiguration('robot_ip')

    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ur_robot_driver'),
            '/launch/ur_control.launch.py',
        ]),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': robot_ip,
            'use_fake_hardware': use_fake_hardware,
            'launch_rviz': 'false',
            'initial_joint_controller': 'joint_trajectory_controller',
        }.items(),
    )

    ur_moveit = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    FindPackageShare('arm_system'),
                    '/launch/ur_moveit_fake.launch.py',
                ]),
                launch_arguments={
                    'ur_type': ur_type,
                    'launch_rviz': launch_rviz,
                    'launch_servo': 'false',
                    'use_sim_time': 'false',
                }.items(),
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('ur_type', default_value='ur5e'),
        DeclareLaunchArgument('use_fake_hardware', default_value='true'),
        DeclareLaunchArgument('robot_ip', default_value='127.0.0.1'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        ur_control,
        ur_moveit,
    ])
