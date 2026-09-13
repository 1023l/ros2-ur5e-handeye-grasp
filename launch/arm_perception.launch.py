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
            DeclareLaunchArgument("detector_backend", default_value="auto"),
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
                        "object_distance_m": 0.45,
                        "publish_hz": 5.0,
                    }
                ],
            ),
        ]
    )
