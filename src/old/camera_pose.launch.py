""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: ee_gripper_link -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "ee_gripper_link",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "-3.45386",
                "--y",
                "0.0081761",
                "--z",
                "0.0595815",
                "--qx",
                "0.600497",
                "--qy",
                "-0.556691",
                "--qz",
                "0.432064",
                "--qw",
                "-0.377914",
                # "--roll",
                # "3.06205",
                # "--pitch",
                # "1.22166",
                # "--yaw",
                # "1.55083",
            ],
        ),
    ]
    return LaunchDescription(nodes)
