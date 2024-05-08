"""
Copyright 2024 SZucchini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file is a modification of repository originally released under the Apache-2.0 License.
Details of the original repository is as follows:
- Original repository: https://github.com/TaatiTeam/MotionAGFormer
- Changes made by: SZucchini
"""
import numpy as np
import torch


def calc_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    eval_idx = [i for i in range(17) if i not in [9, 10]]
    predicted = predicted[:, :, eval_idx, :]
    target = target[:, :, eval_idx, :]
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1), axis=1)


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    eval_idx = [i for i in range(17) if i not in [9, 10]]
    predicted = predicted[:, eval_idx, :]
    target = target[:, eval_idx, :]

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=1)


def spa_mpjpe(predicted, target):
    assert predicted.shape == target.shape
    eval_idx = [i for i in range(17) if i not in [9, 10]]
    predicted = predicted[:, eval_idx, :]
    target = target[:, eval_idx, :]

    # Calculate the mean position across all frames
    muX = np.mean(target, axis=(0, 1), keepdims=True)
    muY = np.mean(predicted, axis=(0, 1), keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    # Calculate the norm across all frames
    normX = np.sqrt(np.sum(X0 ** 2, axis=(0, 1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(0, 1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    # Calculate the correlation matrix, SVD, and rotation matrix once, using all frames
    H = np.matmul(X0.reshape(-1, X0.shape[-1]).T, Y0.reshape(-1, Y0.shape[-1]))
    U, s, Vt = np.linalg.svd(H)
    V = Vt.T
    R = np.matmul(V, U.T)

    # Ensure proper rotations
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = np.matmul(V, U.T)

    # Calculate the scale and translation using the common transformation
    tr = np.sum(s, keepdims=True)
    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    # Apply the common transformation to each frame
    predicted_aligned = a * np.matmul(predicted, R) + t
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=-1), axis=(0, 1))


def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True),
                                dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True),
                             dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)


def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))
