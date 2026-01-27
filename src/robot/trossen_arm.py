"""Trossen arm implementation using trossen_arm SDK."""

import os

import numpy as np
import trossen_arm

from src.robot.arm import RobotArm
from src.utils.log import get_logger
from src.utils.types import Transform4x4

logger = get_logger(__name__)


class TrossenArm(RobotArm):
    """Trossen arm using trossen_arm SDK."""

    def __init__(
        self,
        model: str = "wxai_v0",
        end_effector: str = "wxai_v0_follower",
    ):
        self._ip = os.getenv("ARM_IP", "192.168.1.99")
        self._model = getattr(trossen_arm.Model, model)
        self._end_effector = getattr(trossen_arm.StandardEndEffector, end_effector)
        self._driver = None
        self.connect()

    def connect(self) -> None:
        logger.info(f"Connecting to Trossen arm at {self._ip}...")
        self._driver = trossen_arm.TrossenArmDriver()
        self._driver.configure(
            self._model,
            self._end_effector,
            self._ip,
            False,
        )
        logger.info("Connected")

    def get_ee_pose(self) -> Transform4x4:
        """Get end-effector pose as 4x4 transformation matrix T_base_gripper."""
        # Get [x, y, z, rx, ry, rz] from SDK
        cart = self._driver.get_cartesian_positions()
        x, y, z, rx, ry, rz = cart

        # Build rotation matrix from euler angles (ZYX convention)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        # R = Rz @ Ry @ Rx
        R = np.array([
            [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
            [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
            [-sy,   cy*sx,            cy*cx           ]
        ], dtype=np.float64)

        # Build 4x4 transformation matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T.astype(np.float32)

    def get_gripper_opening(self) -> float:
        """Get gripper opening in meters."""
        output = self._driver.get_robot_output()
        return float(output.joint.gripper.position)

    def get_gripper_effort(self) -> float:
        """Get gripper effort in Newtons."""
        output = self._driver.get_robot_output()
        return float(output.joint.gripper.effort)

    def get_joint_positions(self) -> np.ndarray:
        """Get current arm joint positions (radians)."""
        output = self._driver.get_robot_output()
        return np.array(output.joint.arm.positions)

    def move_down(self, distance: float) -> None:
        """Move end-effector down by distance (meters).

        Approximates Cartesian Z movement by adjusting shoulder joint.
        For small distances (1mm), uses small joint angle increments.
        """
        # Get current joint positions
        current = self.get_joint_positions()

        # Approximate: moving shoulder joint (index 1) forward moves EE down
        # For wxai_v0: ~0.01 rad shoulder change ≈ 5mm Z change (rough estimate)
        angle_delta = distance * 2.0  # radians per meter (tune this)

        new_positions = current.copy()
        new_positions[1] += angle_delta  # Shoulder forward = down

        # Move with short duration for small steps
        self._driver.set_arm_modes(trossen_arm.Mode.position)
        self._driver.set_arm_positions(new_positions, 0.1, True)

    def move_up(self, distance: float) -> None:
        """Move end-effector up by distance (meters)."""
        # Get current joint positions
        current = self.get_joint_positions()

        # Move shoulder backward = up
        angle_delta = distance * 2.0

        new_positions = current.copy()
        new_positions[1] -= angle_delta

        self._driver.set_arm_modes(trossen_arm.Mode.position)
        self._driver.set_arm_positions(new_positions, 0.5, True)

    def go_home(self, duration: float = 2.0) -> None:
        """Go to home position with gripper open."""
        # Home position: all joints at zero
        home_positions = np.zeros(len(self.get_joint_positions()))
        self._driver.set_arm_modes(trossen_arm.Mode.position)
        self._driver.set_arm_positions(home_positions, duration, True)
        self.open_gripper()

    def set_gripper_position(self, position: float, duration: float = 0.5) -> None:
        """Set gripper position in meters."""
        self._driver.set_gripper_mode(trossen_arm.Mode.position)
        self._driver.set_gripper_position(position, duration, True)

    def step_gripper(self, step: float, duration: float = 0.1) -> None:
        """Step gripper by amount (negative = close, positive = open)."""
        current = self.get_gripper_opening()
        target = max(0.0, current + step)
        self._driver.set_gripper_mode(trossen_arm.Mode.position)
        self._driver.set_gripper_position(target, duration, True)

    def open_gripper(self, position: float = 0.04, duration: float = 1.0) -> None:
        """Open gripper to position (default 40mm)."""
        self._driver.set_gripper_mode(trossen_arm.Mode.position)
        self._driver.set_gripper_position(position, duration, True)

    def close_gripper(self) -> None:
        """Close gripper fully."""
        self._driver.set_gripper_mode(trossen_arm.Mode.position)
        self._driver.set_gripper_position(0.0, 1.0, True)

    def set_arm_positions(self, positions: np.ndarray, duration: float = 2.0) -> None:
        """Move arm to joint positions."""
        self._driver.set_arm_modes(trossen_arm.Mode.position)
        self._driver.set_arm_positions(positions, duration, True)

    def close(self) -> None:
        # SDK doesn't have explicit disconnect
        logger.info("Disconnected")
