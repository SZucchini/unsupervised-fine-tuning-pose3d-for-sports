from logging import getLogger, StreamHandler, DEBUG, Formatter

import numpy as np
from scipy.spatial import procrustes

handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger = getLogger("Log")
logger.setLevel(DEBUG)
for h in logger.handlers[:]:
    logger.removeHandler(h)
    h.close()
logger.addHandler(handler)


def calc_scale(gt_kpts, gt_aligned):
    unit = np.linalg.norm(gt_kpts[0, 7, :] - gt_kpts[0, 0, :])
    aligned_unit = np.linalg.norm(gt_aligned[0, 7, :] - gt_aligned[0, 0, :])
    error_scale = unit / aligned_unit
    return error_scale


def aligned_kpts(gt_kpts, est_kpts):
    frame_diff = est_kpts.shape[0] - gt_kpts.shape[0]
    if frame_diff > 0:
        print("Warning: number of frames did not match")
        est_kpts = est_kpts[:-frame_diff, :, :]
    elif frame_diff < 0:
        print("Warning: number of frames did not match")
        gt_kpts = gt_kpts[:frame_diff, :, :]

    # remove the keypoints that definition is slightly different
    gt_kpts = np.delete(gt_kpts, [7, 10], axis=1)
    est_kpts = np.delete(est_kpts, [7, 10], axis=1)
    gt_aligned = np.zeros_like(gt_kpts)
    est_aligned = np.zeros_like(est_kpts)

    for i in range(gt_kpts.shape[0]):
        gt_aligned[i, :, :], est_aligned[i, :, :], _ = procrustes(gt_kpts[i, :, :],
                                                                  est_kpts[i, :, :])

    error_scale = calc_scale(gt_kpts, gt_aligned)
    return gt_aligned, est_aligned, error_scale


def calc_error(gt_aligned, est_aligned, error_scale):
    error = np.sqrt(np.sum((gt_aligned - est_aligned)**2, axis=2))
    error = np.mean(error, axis=1)
    error = np.mean(error, axis=0)
    error = error * error_scale
    return error


def eval_calib(gt_dict, eval_dict):
    eval_calib_result = {}

    for data_name in gt_dict.keys():
        eval_calib_result[data_name] = {}
        eval_calib_result[data_name]['Errors'] = []
        gt_kpts = gt_dict[data_name]  # (N, F, 17, 3)
        calib_kpts = eval_dict[data_name]['3d_kpts_triangulated']  # (N, F, 17, 3)

        for gt_kpt, calib_kpt in zip(gt_kpts, calib_kpts):
            gt_aligned, calib_aligned, error_scale = aligned_kpts(gt_kpt, calib_kpt)
            error = calc_error(gt_aligned, calib_aligned, error_scale)
            eval_calib_result[data_name]['Errors'].append(error)

    for data_name in eval_calib_result.keys():
        mean_error = np.mean(eval_calib_result[data_name]['Errors']) * 1000
        max_error = np.max(eval_calib_result[data_name]['Errors']) * 1000

        logger.debug("Data name: %s", data_name)
        logger.debug("Mean Error: %f [mm]", mean_error)
        logger.debug("Max Error: %f [mm]", max_error)


def eval_mono(gt_dict, eval_dict):
    eval_mono_result = {}

    for data_name in gt_dict.keys():
        eval_mono_result[data_name] = {}
        gt_kpts = gt_dict[data_name]  # (N, F, 17, 3)
        mono_kpts = eval_dict[data_name]['3d_kpts_by_cam']  # (N, 3, F, 17, 3)

        for gt_kpt, mono_kpt in zip(gt_kpts, mono_kpts):
            for cam in range(len(mono_kpt)):
                if cam not in eval_mono_result[data_name].keys():
                    eval_mono_result[data_name][cam] = {}
                    eval_mono_result[data_name][cam]['Errors'] = []

                gt_aligned, mono_aligned, error_scale = aligned_kpts(gt_kpt, mono_kpt[cam])
                error = calc_error(gt_aligned, mono_aligned, error_scale)
                eval_mono_result[data_name][cam]['Errors'].append(error)

    for data_name in eval_mono_result.keys():
        logger.debug("Data name: %s", data_name)
        for cam in eval_mono_result[data_name].keys():
            mean_error = np.mean(eval_mono_result[data_name][cam]['Errors']) * 1000
            max_error = np.max(eval_mono_result[data_name][cam]['Errors']) * 1000

            logger.debug("Camera: %d", cam)
            logger.debug("Mean Error: %f [mm]", mean_error)
            logger.debug("Max Error: %f [mm]", max_error)
