#!/bin/bash

# Trap signals to prevent premature shutdown during demo
trap '' SIGINT SIGTERM

# Ensure clean ROS environment
unset ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=0

# Source ROS
source /opt/ros/jazzy/setup.bash
source /home/skhan61/ros2_ws/install/setup.bash

# Run demo with explicit Python path and unbuffered output
cd /home/skhan61/ros2_ws
exec python3 -u /home/skhan61/ros2_ws/src/trossen_arm_ros/trossen_arm_bringup/demos/single_arm_demo.py
