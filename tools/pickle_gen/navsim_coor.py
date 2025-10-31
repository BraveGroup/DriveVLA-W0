from enum import IntEnum
import numpy as np
import numpy.typing as npt
import math
from dataclasses import dataclass
from typing import Iterable, List, Union




@dataclass
class Point2D:
    """Class to represents 2D points."""

    x: float  # [m] location
    y: float  # [m] location
    __slots__ = "x", "y"

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y)
        """
        return iter((self.x, self.y))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y]
        """
        return np.array([self.x, self.y], dtype=np.float64)

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y))


@dataclass
class StateSE2(Point2D):
    """
    SE2 state - representing [x, y, heading]
    """

    heading: float  # [rad] heading of a state
    __slots__ = "heading"

    @property
    def point(self) -> Point2D:
        """
        Gets a point from the StateSE2
        :return: Point with x and y from StateSE2
        """
        return Point2D(self.x, self.y)

    def as_matrix(self) -> npt.NDArray[np.float32]:
        """
        :return: 3x3 2D transformation matrix representing the SE2 state.
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), self.x],
                [np.sin(self.heading), np.cos(self.heading), self.y],
                [0.0, 0.0, 1.0],
            ]
        )

    def as_matrix_3d(self) -> npt.NDArray[np.float32]:
        """
        :return: 4x4 3D transformation matrix representing the SE2 state projected to SE3.
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), 0.0, self.x],
                [np.sin(self.heading), np.cos(self.heading), 0.0, self.y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )


def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

def convert_absolute_to_relative_se2_array(
    origin: StateSE2, state_se2_array: npt.NDArray[np.float64]
):
    """
    Converts an StateSE2 array from global to relative coordinates.
    :param origin: origin pose of relative coords system
    :param state_se2_array: array of SE2 states with (x,y,θ) in last dim
    :return: SE2 coords array in relative coordinates
    """
    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = state_se2_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])

    return points_rel

def regular_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def convert_relative_to_absolute_se2_trajectory(
    rel_traj: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Convert relative SE2 trajectory to absolute trajectory.
    Assumes initial pose is (0, 0, 0).
    
    :param rel_traj: [N, 3] array of (dx, dy, dtheta) in local frame of previous timestep
    :return: [N, 3] array of absolute (x, y, yaw)
    """
    N = rel_traj.shape[0]
    abs_traj = np.zeros_like(rel_traj)

    x, y, yaw = 0.0, 0.0, 0.0

    for i in range(N):
        dx, dy, dyaw = rel_traj[i]

        # Rotate local motion to global frame
        global_dx = np.cos(yaw) * dx - np.sin(yaw) * dy
        global_dy = np.sin(yaw) * dx + np.cos(yaw) * dy

        # Update global pose
        x += global_dx
        y += global_dy
        yaw = regular_angle(yaw + dyaw)

        abs_traj[i] = [x, y, yaw]

    return abs_traj