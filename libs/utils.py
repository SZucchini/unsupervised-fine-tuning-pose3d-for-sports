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


def get_kpts3d_pixel(kpts3d_camera, camera, root_idx=0):
    """Converts 3D keypoints from camera to pixel coordinates.

    Args:
        kpts3d_camera (ndarray): 3D keypoints in camera coordinates.
        camera (Object): Camera object.
        root_idx (int): Index of the root joint.

    Returns:
        kpts3d_pixel (ndarray): 3D keypoints in pixel coordinates.
        scale_list (ndarray): Scale factors.
    """
    root_joint = kpts3d_camera[:, root_idx, :]
    kpts3d_pixel = np.zeros_like(kpts3d_camera)
    scale_list = []

    for i in range(root_joint.shape[0]):
        tl = root_joint[i, :] - 1000
        br = root_joint[i, :] + 1000
        bbox = np.array([tl, br]).reshape(1, 2, 3)
        bbox = to_cartesian(camera.camera_to_image_space(bbox)).flatten()
        scale = (bbox[2] - bbox[0] + 1) / 2000
        scale_list.append(scale)

        kpts3d_pixel[i, :, :2] = to_cartesian(
            camera.camera_to_image_space(kpts3d_camera)
        )[i, :, :2]
        depth = (kpts3d_camera[i, :, 2] - root_joint[i, 2]) * scale
        kpts3d_pixel[i, :, 2] = depth

    return kpts3d_pixel, np.array(scale_list)


def resample(data_length, target_lenght, replay=False, randomness=True):
    """Resample a sequence of indices.

    Args:
        data_length (int): Original length of the sequence.
        target_lenght (int): Target length of the sequence.
        replay (bool): Whether to allow replay.
        randomness (bool): Whether to add randomness.

    Returns:
        result (ndarray): Resampled sequence of indices.
    """
    if replay:
        if data_length > target_lenght:
            st = np.random.randint(data_length - target_lenght)
            return range(st, st + target_lenght)
        else:
            return np.array(range(target_lenght)) % data_length
    else:
        if randomness:
            even = np.linspace(0, data_length, num=target_lenght, endpoint=False)
            if data_length < target_lenght:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel * low + (1 - sel) * high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape) * interval + even
            result = np.clip(result, a_min=0, a_max=data_length - 1).astype(np.uint32)
        else:
            result = np.linspace(0, data_length, num=target_lenght, endpoint=False, dtype=int)
        return result


def split_clips(data_length, n_frames, data_stride):
    """Split a sequence of indices into clips.

    Args:
        data_length (int): Length of the sequence.
        n_frames (int): Number of frames per clip.
        data_stride (int): Stride between clips.

    Returns:
        result (list): List of clips.
    """
    result = []
    n_clips = 0
    start = 0
    while True:
        if start >= data_length:
            break
        elif start + n_frames > data_length:
            resampled = resample(data_length - start, n_frames) + start
            result.append(resampled)
            n_clips += 1
            break
        else:
            result.append(range(start, start + n_frames))
            start += data_stride
            n_clips += 1
    return result


def split_data(input_2d, label_3d, scale_list, frames, stride):
    """Split data into clips.

    Args:
        input_2d (ndarray): 2D input data.
        label_3d (ndarray): 3D label data.
        scale_list (ndarray): List of scales.
        frames (int): Number of frames per clip.
        stride (int): Stride between clips.

    Returns:
        inputs (ndarray): 2D input data split into clips.
        labels (ndarray): 3D label data split into clips.
        scales (ndarray): List of scales split into clips.
    """
    split_idx = split_clips(input_2d.shape[0], frames, data_stride=stride)
    inputs = input_2d[split_idx]
    labels = label_3d[split_idx]
    scales = scale_list[split_idx]
    return inputs, labels, scales


def normalize_kpts(kpts, w=3840, h=2160):
    """Normalize keypoints to the range [-1, 1].

    Args:
        kpts (ndarray): Keypoints.
        w (int): Image width.
        h (int): Image height.

    Returns:
        kpts_norm (ndarray): Normalized keypoints.
    """
    if kpts.shape[-1] == 2:
        kpts_norm = kpts / w * 2 - [1, h / w]
    elif kpts.shape[-1] == 3:
        kpts_norm = np.zeros_like(kpts)
        kpts_norm[:, :, :2] = kpts[:, :, :2] / w * 2 - [1, h / w]
        kpts_norm[:, :, 2] = kpts[:, :, 2] / w * 2
    return kpts_norm
