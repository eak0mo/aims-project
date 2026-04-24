"""Trajectories"""

from abc import ABC, abstractmethod
import numpy as np


class TaskTrajectory(ABC):
    """Base Operational Space Trajectory implementation

    Inherited classes must implement the following methods:
    - position(t: float) -> np.ndarray
    - velocity(t: float) -> np.ndarray
    - acceleration(t: float) -> np.ndarray
    - rotation(t: float) -> np.ndarray
    - omega(t: float) -> np.ndarray
    - alpha(t: float) -> np.ndarray
    """

    def __init__(self):
        pass

    @abstractmethod
    def position(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def velocity(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def acceleration(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def rotation(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def omega(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def alpha(self, t: float) -> np.ndarray:
        pass


class JointTrajectory(ABC):
    """Base Joint Space Trajectory implementation

    Inherited classes must implement the following methods:
    - joint_positions(t: float) -> np.ndarray
    - joint_velocities(t: float) -> np.ndarray
    - joint_accelerations(t: float) -> np.ndarray
    """

    def __init__(self):
        pass

    @abstractmethod
    def joint_positions(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def joint_velocities(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def joint_accelerations(self, t: float) -> np.ndarray:
        pass


class SinusoidalTaskTrajectory(TaskTrajectory):
    """An example sinusoidal task-space position trajectory for the robot to follow

    Args:
        init_pos (np.ndarray): Initial position of the end-effector, shape (3,)
        init_rot (np.ndarray): Initial rotation of the end-effector, shape (3, 3)
        amplitude (np.ndarray): X,Y,Z amplitudes of the sinusoid, shape (3,)
        angular_freq (np.ndarray): X,Y,Z angular frequencies of the sinusoid, shape (3,)
        phase (np.ndarray): X,Y,Z phase offsets of the sinusoid, shape (3,)
    """

    def __init__(
        self,
        init_pos: np.ndarray,
        init_rot: np.ndarray,
        amplitude: np.ndarray,
        angular_freq: np.ndarray,
        phase: np.ndarray,
    ):
        self.init_pos = np.asarray(init_pos)
        self.init_rot = np.asarray(init_rot)
        self.amplitude = np.asarray(amplitude)
        self.angular_freq = np.asarray(angular_freq)
        self.phase = np.asarray(phase)

        assert self.init_pos.shape == (3,)
        assert self.init_rot.shape == (3, 3)
        assert self.amplitude.shape == (3,)
        assert self.angular_freq.shape == (3,)
        assert self.phase.shape == (3,)

    # Simple sinusoidal positional trajectory

    def position(self, t: float) -> np.ndarray:
        return self.init_pos + self.amplitude * np.sin(
            self.angular_freq * t + self.phase
        )

    def velocity(self, t: float) -> np.ndarray:
        return (
            self.amplitude
            * self.angular_freq
            * np.cos(self.angular_freq * t + self.phase)
        )

    def acceleration(self, t: float) -> np.ndarray:
        return (
            -self.amplitude
            * self.angular_freq**2
            * np.sin(self.angular_freq * t + self.phase)
        )

    # Maintain a fixed orientation

    def rotation(self, t: float) -> np.ndarray:
        return self.init_rot

    def omega(self, t: float) -> np.ndarray:
        return np.zeros(3)

    def alpha(self, t: float) -> np.ndarray:
        return np.zeros(3)
