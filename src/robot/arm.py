"""Robot arm base class."""

from abc import ABC, abstractmethod

import numpy as np

from src.utils.types import Transform4x4


class RobotArm(ABC):
    """Abstract base class for robot arms."""

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def get_ee_pose(self) -> Transform4x4:
        pass

    @abstractmethod
    def get_gripper_opening(self) -> float:
        pass

    @abstractmethod
    def move_down(self, distance: float) -> None:
        pass

    @abstractmethod
    def move_up(self, distance: float) -> None:
        pass

    @abstractmethod
    def open_gripper(self) -> None:
        pass

    @abstractmethod
    def close_gripper(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
