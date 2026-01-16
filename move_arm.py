"""
Simple script to move the Trossen Arm.
"""

import time
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

# Arm configuration
ARM_IP = "192.168.1.99"

def main():
    # Create driver
    driver = TrossenArmDriver()

    print(f"Connecting to arm at {ARM_IP}...")
    driver.configure(
        model=Model.wxai_v0,
        end_effector=StandardEndEffector.wxai_v0_leader,
        serv_ip=ARM_IP,
        clear_error=True,
        timeout=10.0
    )
    print("Connected!")

    # Get current joint positions
    positions = driver.get_all_positions()
    print(f"Current joint positions: {list(positions)}")

    # Set all joints to position mode
    driver.set_all_modes(Mode.position)

    # Move to home position (all zeros)
    print("Moving to home position...")
    num_joints = driver.get_num_joints()
    home_position = [0.0] * num_joints
    driver.set_all_positions(home_position, goal_time=2.0)

    # Small wave motion on joint 1
    print("Performing small wave motion on joint 1...")
    for i in range(3):
        # Move joint 1 slightly
        positions = [0.0] * num_joints
        positions[0] = 0.3  # Move first joint
        driver.set_all_positions(positions, goal_time=0.5)

        positions[0] = -0.3
        driver.set_all_positions(positions, goal_time=0.5)

    # Return to home
    print("Returning to home position...")
    driver.set_all_positions([0.0] * num_joints, goal_time=1.0)

    print("Done!")
    driver.cleanup()

if __name__ == "__main__":
    main()
