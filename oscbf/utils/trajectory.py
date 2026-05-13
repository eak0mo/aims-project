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


class WaypointTaskTrajectory(TaskTrajectory):
    """A piecewise linear task-space position trajectory moving through specified waypoints.

    Useful for multi-stage tasks like pick-and-place or pick-and-drop sequences.

    Args:
        waypoints (np.ndarray): Array of target positions, shape (N, 3)
        times (np.ndarray): Array of timestamps corresponding to each waypoint, shape (N,)
        init_rot (np.ndarray): Fixed rotation matrix of the end-effector, shape (3, 3)
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        times: np.ndarray,
        init_rot: np.ndarray,
    ):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.init_rot = np.asarray(init_rot, dtype=float)

        assert self.waypoints.ndim == 2 and self.waypoints.shape[1] == 3, "waypoints must have shape (N, 3)"
        assert self.times.ndim == 1, "times must be a 1D array"
        assert len(self.waypoints) == len(self.times), "Number of waypoints must match number of times"
        assert self.init_rot.shape == (3, 3), "init_rot must have shape (3, 3)"
        assert np.all(np.diff(self.times) > 0), "times must be strictly increasing"

    def position(self, t: float) -> np.ndarray:
        if t <= self.times[0]:
            return self.waypoints[0]
        if t >= self.times[-1]:
            return self.waypoints[-1]

        # Find the segment [i, i+1] containing t
        idx = np.searchsorted(self.times, t, side="right") - 1
        t0, t1 = self.times[idx], self.times[idx + 1]
        p0, p1 = self.waypoints[idx], self.waypoints[idx + 1]

        return p0 + (p1 - p0) * ((t - t0) / (t1 - t0))

    def velocity(self, t: float) -> np.ndarray:
        if t < self.times[0] or t >= self.times[-1]:
            return np.zeros(3)

        idx = np.searchsorted(self.times, t, side="right") - 1
        t0, t1 = self.times[idx], self.times[idx + 1]
        p0, p1 = self.waypoints[idx], self.waypoints[idx + 1]

        return (p1 - p0) / (t1 - t0)

    def acceleration(self, t: float) -> np.ndarray:
        return np.zeros(3)

    def rotation(self, t: float) -> np.ndarray:
        return self.init_rot

    def omega(self, t: float) -> np.ndarray:
        return np.zeros(3)

    def alpha(self, t: float) -> np.ndarray:
        return np.zeros(3)

