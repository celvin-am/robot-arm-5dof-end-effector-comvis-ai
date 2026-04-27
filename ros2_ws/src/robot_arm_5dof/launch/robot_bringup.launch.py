"""
robot_bringup.launch.py — Phase 1 stub launch file for robot_arm_5dof.

This is a placeholder. The real bringup launch will be built in Phase 9
once all nodes are implemented.

Usage:
  ros2 launch robot_arm_5dof robot_bringup.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description with placeholder node."""
    # Placeholder node — will be replaced with actual nodes in later phases.
    # The entry point robot_arm_5dof_node exists as a stub in Phase 1
    # and will be replaced by real nodes (camera_yolo_node, etc.) in Phase 7+.
    placeholder_node = Node(
        package='robot_arm_5dof',
        executable='robot_arm_5dof_node',
        name='robot_arm_5dof_node',
        output='screen',
        parameters=[],
    )

    return LaunchDescription([
        placeholder_node,
    ])