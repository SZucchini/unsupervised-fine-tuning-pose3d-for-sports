import numpy as np
from pycalib.calib import rebase_all

H36M_KEY = {
    "Pelvis": 0,
    "RHip": 1,
    "RKnee": 2,
    "RAnkle": 3,
    "LHip": 4,
    "LKnee": 5,
    "LAnkle": 6,
    "Spine": 7,
    "Thorax": 8,
    "Nose": 9,
    "Head": 10,
    "LShoulder": 11,
    "LElbow": 12,
    "LWrist": 13,
    "RShoulder": 14,
    "RElbow": 15,
    "RWrist": 16,
}

H36M_BONE = np.array(
    [
        [H36M_KEY["Head"], H36M_KEY["Nose"]],
        [H36M_KEY["Nose"], H36M_KEY["Thorax"]],
        [H36M_KEY["Thorax"], H36M_KEY["Spine"]],
        [H36M_KEY["Thorax"], H36M_KEY["RShoulder"]],
        [H36M_KEY["Thorax"], H36M_KEY["LShoulder"]],
        [H36M_KEY["RShoulder"], H36M_KEY["RElbow"]],
        [H36M_KEY["LShoulder"], H36M_KEY["LElbow"]],
        [H36M_KEY["RWrist"], H36M_KEY["RElbow"]],
        [H36M_KEY["LWrist"], H36M_KEY["LElbow"]],
        [H36M_KEY["Spine"], H36M_KEY["Pelvis"]],
        [H36M_KEY["RHip"], H36M_KEY["Pelvis"]],
        [H36M_KEY["LHip"], H36M_KEY["Pelvis"]],
        [H36M_KEY["RHip"], H36M_KEY["RKnee"]],
        [H36M_KEY["RKnee"], H36M_KEY["RAnkle"]],
        [H36M_KEY["LHip"], H36M_KEY["LKnee"]],
        [H36M_KEY["LKnee"], H36M_KEY["LAnkle"]],
    ],
    dtype=np.int64,
)


def invRT_batch(R_w2c_gt, t_w2c_gt):
    t_c2w_gt = []
    R_c2w_gt = []

    if len(t_w2c_gt.shape) == 2:
        t_w2c_gt = t_w2c_gt[:, :, None]

    for R_w2c_gt_i, t_w2c_gt_i in zip(R_w2c_gt, t_w2c_gt):
        R_c2w_gt_i, t_c2w_gt_i = invRT(R_w2c_gt_i, t_w2c_gt_i)
        R_c2w_gt.append(R_c2w_gt_i)
        t_c2w_gt.append(t_c2w_gt_i)

    t_c2w_gt = np.array(t_c2w_gt)
    R_c2w_gt = np.array(R_c2w_gt)

    return R_c2w_gt, t_c2w_gt


def invRT(R, t):
    T = np.eye(4)
    if t.shape == (3, 1):
        t = t[:, -1]

    T[:3, :3] = R
    T[:3, 3] = t
    invT = np.linalg.inv(T)
    invR = invT[0:3, 0:3]
    invt = invT[0:3, 3]
    return invR, invt


def constraint_mat_from_single_view(p, proj_mat):
    u, v = p
    const_mat = np.empty((2, 4))
    const_mat[0, :] = u * proj_mat[2, :] - proj_mat[0, :]
    const_mat[1, :] = v * proj_mat[2, :] - proj_mat[1, :]
    return const_mat[:, :3], -const_mat[:, 3]


def constraint_mat(p_stack, proj_mat_stack):
    lhs_list = []
    rhs_list = []
    for p, proj in zip(p_stack, proj_mat_stack):
        lhs, rhs = constraint_mat_from_single_view(p, proj)
        lhs_list.append(lhs)
        rhs_list.append(rhs)
    A = np.vstack(lhs_list)
    b = np.hstack(rhs_list)
    return A, b


def triangulate_point(p_stack, proj_mat_stack, confs=None):
    A, b = constraint_mat(p_stack, proj_mat_stack)
    if confs is None:
        confs = np.ones(b.shape)
    else:
        confs = np.array(confs).repeat(2)

    p_w, _, rank, _ = np.linalg.lstsq(A * confs.reshape((-1, 1)), b * confs, rcond=None)
    if np.sum(confs > 0) <= 2:
        return np.full((3), np.nan)

    if rank < 3:
        raise Exception("not enough constraint")
    return p_w


def triangulate_with_conf(p2d, s2d, K, R_w2c, t_w2c, mask):
    """
    p2d   : np.ndarray, (C, F, 17, 2)
    s2d   : np.ndarray, (C, F, 17)
    K     : np.ndarray, (C, 3, 3)
    R_w2c : np.ndarray, (C, 3, 3)
    t_w2c : np.ndarray, (C, 3, 1)
    """
    R_w2c, t_w2c = rebase_all(R_w2c, t_w2c, normalize_scaling=True)

    assert p2d.ndim == 4
    assert s2d.ndim == 3

    Nc, Nf, Nj, _ = p2d.shape
    P_est = []
    for i in range(Nc):
        P_est.append(K[i] @ np.hstack((R_w2c[i], t_w2c[i])))
    P_est = np.array(P_est)

    X = []
    for i in range(Nf):
        for j in range(Nj):
            x = p2d[:, i, j, :].reshape((Nc, 2))
            m = mask[:, i, j]
            confi = s2d[:, i, j]

            if confi.sum() > 0 and m.sum() > 1:
                x3d = triangulate_point(x[m], P_est[m], confi[m])
            else:
                x3d = np.full(4, np.nan)
            X.append(x3d[:3])

    X = np.array(X)
    X = X.reshape(Nf, Nj, 3)
    return X
