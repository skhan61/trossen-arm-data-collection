"""
Initialize and configure the Trossen Arm to home position.
"""

from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

ARM_IP = "192.168.1.99"

def main():
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

    # Print arm info
    print(f"\n=== Arm Info ===")
    print(f"Controller version: {driver.get_controller_version()}")
    print(f"Driver version: {driver.get_driver_version()}")
    print(f"Number of joints: {driver.get_num_joints()}")

    # Get current positions
    print(f"\n=== Current State ===")
    positions = driver.get_all_positions()
    print(f"Joint positions (rad): {[round(p, 4) for p in positions]}")

    velocities = driver.get_all_velocities()
    print(f"Joint velocities: {[round(v, 4) for v in velocities]}")

    # Get joint limits
    print(f"\n=== Joint Limits ===")
    limits = driver.get_joint_limits()
    for i, limit in enumerate(limits):
        print(f"Joint {i}: pos=[{limit.position_min:.2f}, {limit.position_max:.2f}] rad")

    # Set to position mode
    print(f"\n=== Setting to Position Mode ===")
    driver.set_all_modes(Mode.position)
    print("All joints set to position mode")

    # Move to home position (all zeros)
    print(f"\n=== Moving to Home Position ===")
    num_joints = driver.get_num_joints()
    home_position = [0.0] * num_joints
    print(f"Target home position: {home_position}")
    driver.set_all_positions(home_position, goal_time=3.0)
    print("Arm is now at home position!")

    # Final positions
    final_positions = driver.get_all_positions()
    print(f"Final positions: {[round(p, 4) for p in final_positions]}")

    driver.cleanup()
    print("\nDone!")

if __name__ == "__main__":
    main()
