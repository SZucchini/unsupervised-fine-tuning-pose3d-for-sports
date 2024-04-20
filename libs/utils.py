"""
Camera class is edited from the original code in the following link:
https://github.com/anibali/aspset-510/blob/master/src/aspset510/camera.py
"""
import numpy as np


def to_cartesian(points):
    """Converts points from homogeneous to cartesian coordinates.

    Args:
        points (ndarray): Points in homogeneous coordinates.

    Returns:
        ndarray: Points in cartesian coordinates.
    """
    return points[..., :-1] / points[..., -1:]


def to_homogeneous(points):
    """Converts points from cartesian to homogeneous coordinates.

    Args:
        points (ndarray): Points in cartesian coordinates.

    Returns:
        ndarray: Points in homogeneous coordinates.
    """
    return np.concatenate([points, np.ones_like(points[..., -1:])], -1)


def ensure_homogeneous(points, d):
    """Ensures that points are in homogeneous coordinates.

    Args:
        points (ndarray): Points in cartesian or homogeneous coordinates.
        d (int): Dimension of the points.

    Returns:
        ndarray: Points in homogeneous coordinates.
    """
    if points.shape[-1] == d + 1:
        return points
    assert points.shape[-1] == d
    return to_homogeneous(points)


class Camera:
    """
    Represents a camera in 3D space.
    """
    def __init__(self, intrinsic, extrinsic):
        """Initializes a camera.

        Args:
            intrinsic (ndarray): Intrinsic camera parameters.
            extrinsic (ndarray): Extrinsic camera parameters.
            K (ndarray): Intrinsic camera matrix.
            R (ndarray): Rotation matrix.
            t (ndarray): Translation vector.
        """
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.K = intrinsic[:, :3]
        self.R = extrinsic[:3, :3]
        self.t = extrinsic[:3, 3]

    @property
    def projection_matrix(self):
        return self.intrinsic @ self.extrinsic

    def world_to_camera_space(self, points_3d):
        """Transform points from 3D world space to 3D camera space.

        Args:
            points_3d: 3D points in world space.

        Returns:
            Homogeneous 3D points in camera space.
        """
        points_3d = ensure_homogeneous(points_3d, d=3)
        return points_3d @ self.extrinsic_matrix.T

    def camera_to_image_space(self, points_3d):
        """Transform points from 3D camera space to 2D image space.

        Args:
            points_3d: 3D points in camera space.

        Returns:
            Homogeneous 2D points in image space.
        """
        points_3d = ensure_homogeneous(points_3d, d=3)
        return points_3d @ self.intrinsic_matrix.T

    def world_to_image_space(self, points_3d):
        """Transform points from 3D world space to 2D image space.

        Args:
            points_3d: 3D points in world space.

        Returns:
            Homogeneous 2D points in image space.
        """
        points_3d = ensure_homogeneous(points_3d, d=3)
        return points_3d @ self.projection_matrix.T
