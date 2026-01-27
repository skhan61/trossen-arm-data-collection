"""Trossen arm interface.

TODO: Implement with actual robot SDK when hardware available.
"""

from src.utils.log import get_logger
from src.utils.types import Transform4x4

logger = get_logger(__name__)


class RobotArm:
    """Trossen arm interface.

    TODO: Implement with actual robot SDK.
    """

    def __init__(self):
        logger.info("Connecting to robot arm...")
        # TODO: Initialize robot connection
        raise NotImplementedError("Robot connection not implemented")

    def get_ee_pose(self) -> Transform4x4:
        """Get current end-effector pose in base frame."""
        raise NotImplementedError

    def get_gripper_opening(self) -> float:
        """Get gripper opening in meters."""
        raise NotImplementedError

    def move_to(self, pose: Transform4x4) -> None:
        """Move end-effector to target pose."""
        raise NotImplementedError

    def move_down(self, distance: float) -> None:
        """Move end-effector down by distance (meters)."""
        raise NotImplementedError

    def move_up(self, distance: float) -> None:
        """Move end-effector up by distance (meters)."""
        raise NotImplementedError

    def close(self) -> None:
        """Disconnect from robot."""
        raise NotImplementedError
