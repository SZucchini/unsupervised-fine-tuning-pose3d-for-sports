import cv2
import numpy as np
from scipy.optimize import least_squares

from .utils import H36M_BONE, triangulate_with_conf


def to_theta(R, t, x):
    rvecs = np.array([cv2.Rodrigues(r)[0].flatten() for r in R]).flatten()
    return np.hstack([rvecs, t.flatten(), x.flatten()])


def from_theta(theta, C):
    rvecs = theta[0:3*C].reshape((C, 3))
    tvecs = theta[3*C:6*C].reshape((C, 3, 1))
    x = theta[6*C:].reshape((-1, 3))
    R = np.array([cv2.Rodrigues(r)[0] for r in rvecs])
    return R, tvecs, x


def objfun_nll(K_all, R_all, t_all, x_all, y_all, y_mask_all, s_all):
    e_all = []
    for K, R, t, mask, y, s in zip(K_all, R_all, t_all, y_mask_all, y_all, s_all):
        x = x_all[mask]
        y = y[mask]
        s = np.sqrt(2) * s[mask]
        s = np.copy(s.reshape((-1, 1)))

        t = np.copy(t)
        y_hat = K @ (R @ x.T + t.reshape((3, 1)))
        y_hat = (y_hat[:2, :] / y_hat[2, :]).T
        e = y - y_hat[:, :2]
        e = e * s
        e_all.append(e.flatten())

    return np.concatenate(e_all)


def objfun_var3d(R_all, vc_all, vc_mask_all, bone_idx):
    C = len(R_all)

    vw = []
    for R, vc in zip(R_all, vc_all):
        vw.append(vc @ R)
    vw = np.array(vw)
    vw[~vc_mask_all] = np.nan

    bones = vw[:, :, bone_idx, :]
    dirs = bones[:, :, :, 0, :] - bones[:, :, :, 1, :]
    dirs = dirs.reshape(C, -1, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=2)[:, :, None]

    m_invalid = np.isnan(dirs).any(axis=(0, 2))
    var = 1 - np.linalg.norm(np.nanmean(dirs[:, ~m_invalid], axis=0), axis=1)

    return var


def objfun_varbone(x_all, bone_idx):
    bone = x_all[:, bone_idx, :]
    bone_length = np.linalg.norm(bone[:, :, 0, :] - bone[:, :, 1, :], axis=2)
    bone_var = np.var(bone_length, axis=0)

    return bone_var


def objfun(params, K, sp2d, ss2d, sp3d, ss3d, bone_idx, C, N, J, lambda1, lambda2):
    R_w2c, t_w2c, x = from_theta(params, C)
    E = []
    e = objfun_nll(
        K,
        R_w2c,
        t_w2c,
        x,
        sp2d,
        (ss2d > 0).reshape((C, N * J)),
        ss2d.reshape((C, N * J)),
    )

    E.append(e.flatten())
    e = objfun_var3d(R_w2c, sp3d, (ss3d > 0), bone_idx)
    E.append(e * lambda1)

    e = objfun_varbone(x.reshape(N, J, 3), bone_idx)
    E.append(e * lambda2)

    return np.concatenate(E)


def bandle_adjust(cfg, R_w2c, t_w2c, sp2d, ss2d, sp3d, ss3d, K):
    lambda1 = cfg.CALIB.BA_LAMBDA1
    lambda2 = cfg.CALIB.BA_LAMBDA2

    C = sp2d.shape[0]
    N = sp2d.shape[1]
    J = sp2d.shape[2]

    x_all = triangulate_with_conf(sp2d, ss2d, K, R_w2c, t_w2c, (ss2d > 0))
    x_all = x_all.reshape(N * J, 3)

    assert x_all.shape == (N * J, 3)

    theta0 = to_theta(R_w2c, t_w2c, x_all)
    res = least_squares(
        objfun,
        theta0,
        verbose=True,
        ftol=1e-4,
        method="trf",
        args=(
            K,
            sp2d.reshape((C, N * J, 2)),
            ss2d,
            sp3d,
            ss3d,
            H36M_BONE,
            C,
            N,
            J,
            lambda1,
            lambda2,
        ),
    )

    R_est, t_est, x_opt = from_theta(res["x"], C)
    kpts3d_tri = triangulate_with_conf(sp2d, ss2d,
                                       K, R_est, t_est, (ss2d > 0))
    return R_est, t_est, x_opt, kpts3d_tri
