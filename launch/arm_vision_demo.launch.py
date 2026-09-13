"""Perception + hand-eye + fake gripper (optional YOLO)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_synthetic", default_value="true"),
            DeclareLaunchArgument("target_color", default_value="red"),
            # auto|yolo|color — auto uses YOLO when models/yolov8n.pt exists
            DeclareLaunchArgument("detector_backend", default_value="auto"),
            DeclareLaunchArgument("yolo_model_path", default_value=""),
            DeclareLaunchArgument(
                "yolo_classes",
                default_value="person,bottle,cup,sports ball,bus,car",
            ),
            DeclareLaunchArgument("scene_mode", default_value="hard"),
            DeclareLaunchArgument("use_apparent_size", default_value="true"),
            Node(
                package="arm_system",
                executable="arm_perception.py",
                name="arm_perception",
                output="screen",
                parameters=[
                    {
                        "use_synthetic": ParameterValue(
                            LaunchConfiguration("use_synthetic"), value_type=bool
                        ),
                        "target_color": LaunchConfiguration("target_color"),
                        "detector_backend": LaunchConfiguration("detector_backend"),
                        "yolo_model_path": LaunchConfiguration("yolo_model_path"),
                        "yolo_classes": LaunchConfiguration("yolo_classes"),
                        "scene_mode": LaunchConfiguration("scene_mode"),
                        "use_apparent_size": ParameterValue(
                            LaunchConfiguration("use_apparent_size"), value_type=bool
                        ),
                        "object_distance_m": 0.45,
                        "object_size_m": 0.06,
                        "publish_hz": 5.0,
                    }
                ],
            ),
            Node(
                package="arm_system",
                executable="arm_handeye.py",
                name="arm_handeye",
                output="screen",
            ),
            Node(
                package="arm_system",
                executable="arm_gripper.py",
                name="arm_fake_gripper",
                output="screen",
            ),
        ]
    )
