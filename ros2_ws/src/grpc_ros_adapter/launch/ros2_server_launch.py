from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('debug', default_value='false'),
        DeclareLaunchArgument('server_ip', default_value='0.0.0.0'),
        DeclareLaunchArgument('server_port', default_value='30052'),
        DeclareLaunchArgument('topic_prefix', default_value=''),
        DeclareLaunchArgument('lockstep_enabled', default_value='false'),
        DeclareLaunchArgument(
            'lockstep_ack_topic',
            default_value='/sim/motion_controller/debug/control_tick_complete',
        ),
        DeclareLaunchArgument('lockstep_ack_timeout_wall_sec', default_value='15.0'),
        Node(
            package='grpc_ros_adapter',
            executable='server',
            namespace='grpc_ros_adapter',
            name='grpc_ros_adapter',
            parameters=[
            {
                "server_ip" : LaunchConfiguration('server_ip'),
                "server_port": LaunchConfiguration('server_port'),
                "topic_prefix": LaunchConfiguration('topic_prefix'),
                "lockstep_enabled": LaunchConfiguration('lockstep_enabled'),
                "lockstep_ack_topic": LaunchConfiguration('lockstep_ack_topic'),
                "lockstep_ack_timeout_wall_sec": LaunchConfiguration('lockstep_ack_timeout_wall_sec'),
            }]
        )
    ])
