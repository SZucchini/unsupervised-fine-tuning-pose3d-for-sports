"""
See the License for the specific language governing permissions and limitations under those Licenses.

Original and modified code from:
- kyotovision-public/extrinsic-camera-calibration-from-a-moving-person (evaluate_extrinsic, eval_R, eval_t, and align) under MIT License: https://github.com/kyotovision-public/extrinsic-camera-calibration-from-a-moving-person
"""
import numpy as np
from pycalib.calib import absolute_orientation

from common.utils import invRT_batch
from .loss import calc_mpjpe, p_mpjpe, spa_mpjpe
from .utils import Camera, get_kpts3d_pixel


def align(pred_R, pred_t, gt_R, gt_t):
    assert np.all(gt_R.shape == pred_R.shape)
    assert np.all(gt_t.shape == pred_t.shape)
    R1 = gt_R
    R2 = pred_R
    t1 = gt_t
    t2 = pred_t
    p1 = []
    p2 = []
    for i in range(len(pred_R)):
        p1.append(-R1[i].T @ t1[i])
        p2.append(-R2[i].T @ t2[i])
    p1 = np.array(p1).reshape((-1, 3)).T
    p2 = np.array(p2).reshape((-1, 3)).T

    R, t, s = absolute_orientation(p2, p1)

    R2n = []
    t2n = []
    Rc2w = []
    for i in range(len(R1)):
        R2n.append(R2[i] @ R.T)
        Rc2w.append((R @ R2[i].T))
        x = -R2[i].T @ t2[i]
        x = s * R @ x + t
        t2n.append((s * R @ (-R2[i].T @ t2[i]) + t.reshape(3, 1)))
    R2n = np.array(R2n)
    t2n = np.array(t2n)
    Rc2w = np.array(Rc2w)
    Rw2c, tw2c = invRT_batch(Rc2w, t2n)

    return Rw2c, tw2c[:, :, None]


def eval_R(R1, R2):
    def thetaR(R):
        e = (np.trace(R) - 1.0) / 2.0
        if e > 1:
            e = 1
        elif e < -1:
            e = -1
        return np.arccos(e)
    return np.linalg.norm(thetaR(R1.T @ R2)) / np.sqrt(2)


def eval_t(t1, t2):
    return np.linalg.norm(t1.flatten() - t2.flatten())


def evaluate_extrinsic(pred_R, pred_t, gt_R, gt_t):
    alined_R, alined_t = align(pred_R, pred_t, gt_R, gt_t)

    error_R, error_t = 0, 0
    for i in range(len(alined_R)):
        error_R += eval_R(alined_R[i], gt_R[i])
        error_t += eval_t(alined_t[i], gt_t[i])

    return error_R / len(alined_R), error_t / len(alined_t)


def evaluate_plabels(tri_kpts3d, pred_R, pred_t, K, gt_kpts3d, gt_pixel, gt_pixel_scale):
    pa_mpjpe = p_mpjpe(tri_kpts3d, gt_kpts3d)
    spa_mpjpe_error = spa_mpjpe(tri_kpts3d, gt_kpts3d)

    cameras = []
    for i in range(len(K)):
        intrinsic = np.hstack((K[i], np.zeros((3, 1))))
        pred_Rt = np.hstack((pred_R[i], pred_t[i]))
        extrinsic = np.vstack((pred_Rt, np.array([0, 0, 0, 1])))
        cameras.append(Camera(intrinsic, extrinsic))

    kpts3d_cameras = []
    for i in range(len(cameras)):
        kpts3d_cameras.append(cameras[i].world_to_camera_space(tri_kpts3d)[:, :, :3])

    pred_kpts3d_pixel = []
    for i, kpts3d_camera in enumerate(kpts3d_cameras):
        kpts3d_pixel, _ = get_kpts3d_pixel(kpts3d_camera, cameras[i])
        pred_kpts3d_pixel.append(kpts3d_pixel)

    mpjpe_list = []
    for i in range(len(pred_kpts3d_pixel)):
        for j in range(pred_kpts3d_pixel[i].shape[0]):
            pred_kpts3d_pixel[i][j, :, :] -= pred_kpts3d_pixel[i][j, 0, :]
            gt_pixel[i][j, :, :] -= gt_pixel[i][j, 0, :]

        pred_kpts3d_pixel[i] = pred_kpts3d_pixel[i] / gt_pixel_scale[i][:, None, None]
        gt_pixel[i] = gt_pixel[i] / gt_pixel_scale[i][:, None, None]

        mpjpe = calc_mpjpe(pred_kpts3d_pixel[i].reshape(1, *pred_kpts3d_pixel[i].shape),
                           gt_pixel[i].reshape(1, *gt_pixel[i].shape))
        mpjpe_list.append(mpjpe)

    return np.mean(mpjpe_list), np.mean(pa_mpjpe), spa_mpjpe_error
