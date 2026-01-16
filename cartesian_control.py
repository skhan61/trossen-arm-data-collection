"""
Cartesian position control for Trossen Arm.
Moves the end effector in 3D space (X, Y, Z + rotations).
"""

import trossen_arm

ARM_IP = "192.168.1.99"

def main():
    # Initialize the driver
    driver = trossen_arm.TrossenArmDriver()

    print(f"Connecting to arm at {ARM_IP}...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        ARM_IP,
        True  # clear_error
    )
    print("Connected!")

    # Set the arm joints to position mode
    driver.set_arm_modes(trossen_arm.Mode.position)

    # Get the current Cartesian positions [x, y, z, rx, ry, rz]
    cartesian_positions = driver.get_cartesian_positions()
    print(f"Current Cartesian position: {[round(p, 3) for p in cartesian_positions]}")
    print("  X (forward/back): {:.3f}m".format(cartesian_positions[0]))
    print("  Y (left/right):   {:.3f}m".format(cartesian_positions[1]))
    print("  Z (up/down):      {:.3f}m".format(cartesian_positions[2]))

    # Store starting position
    start_position = list(cartesian_positions)

    # Move the arm up by 0.05m
    print("\nMoving up by 0.05m...")
    cartesian_positions[2] += 0.05
    driver.set_cartesian_positions(
        cartesian_positions,
        trossen_arm.InterpolationSpace.cartesian
    )

    # Move the end effector left by 0.05m
    print("Moving left by 0.05m...")
    cartesian_positions[1] += 0.05
    driver.set_cartesian_positions(
        cartesian_positions,
        trossen_arm.InterpolationSpace.cartesian
    )

    # Move the end effector forward by 0.05m
    print("Moving forward by 0.05m...")
    cartesian_positions[0] += 0.05
    driver.set_cartesian_positions(
        cartesian_positions,
        trossen_arm.InterpolationSpace.cartesian
    )

    # Rotate the end effector about the z-axis by 0.3 rad
    print("Rotating around Z by 0.3 rad...")
    cartesian_positions[5] += 0.3
    driver.set_cartesian_positions(
        cartesian_positions,
        trossen_arm.InterpolationSpace.cartesian
    )

    # Return to start position
    print("\nReturning to start position...")
    driver.set_cartesian_positions(
        start_position,
        trossen_arm.InterpolationSpace.cartesian
    )

    # Final position
    final_pos = driver.get_cartesian_positions()
    print(f"Final Cartesian position: {[round(p, 3) for p in final_pos]}")

    print("\nDone!")

if __name__ == "__main__":
    main()
