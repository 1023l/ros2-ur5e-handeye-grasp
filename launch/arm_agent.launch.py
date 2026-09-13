from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_llm", default_value="false"),
            DeclareLaunchArgument("llm_api_key", default_value=""),
            DeclareLaunchArgument("llm_api_base", default_value="https://api.openai.com/v1"),
            DeclareLaunchArgument("llm_model", default_value="gpt-4o-mini"),
            Node(
                package="arm_system",
                executable="arm_agent.py",
                name="arm_agent",
                output="screen",
                parameters=[
                    {
                        "use_llm": ParameterValue(
                            LaunchConfiguration("use_llm"), value_type=bool
                        ),
                        "llm_api_key": LaunchConfiguration("llm_api_key"),
                        "llm_api_base": LaunchConfiguration("llm_api_base"),
                        "llm_model": LaunchConfiguration("llm_model"),
                    }
                ],
            ),
        ]
    )
