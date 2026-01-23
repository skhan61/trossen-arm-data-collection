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
                "-0.245197",
                "--y",
                "0.0169362",
                "--z",
                "-0.00274736",
                "--qx",
                "-0.165935",
                "--qy",
                "0.00604235",
                "--qz",
                "-0.0122625",
                "--qw",
                "0.986042",
                # "--roll",
                # "2.80833",
                # "--pitch",
                # "3.12561",
                # "--yaw",
                # "3.11941",
            ],
        ),
    ]
    return LaunchDescription(nodes)
