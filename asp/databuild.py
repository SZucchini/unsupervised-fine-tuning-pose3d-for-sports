import glob
import os
import pickle
from logging import getLogger, StreamHandler, DEBUG, Formatter

import numpy as np

from . import _init_paths
from common.calibration import calibrate
from .evaluation import evaluate_extrinsic, evaluate_plabels
from .infernece import estimate3d
from .utils import Camera, split_data

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


def to_pixel_coordinate(tri_kpts3d, pred_R, pred_t, K):
    cameras = []
    for i in range(len(K)):
        intrinsic = np.hstack((K[i], np.zeros((3, 1))))
        pred_Rt = np.hstack((pred_R[i], pred_t[i]))
        extrinsic = np.vstack((pred_Rt, np.array([0, 0, 0, 1])))
        cameras.append(Camera(intrinsic, extrinsic))

    kpts3d_cameras = []
    for i in range(len(cameras)):
        kpts3d_cameras.append(cameras[i].world_to_camera_space(tri_kpts3d)[:, :, :3])

    return np.array(kpts3d_cameras)


def generate_plabel_dataset(cfg, model, itr):
    train_path = os.path.join(cfg.WORKSPACE, f"dataset/train{itr}")
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    data_path = cfg.DATA.PSEUDO_PATH
    data_files = glob.glob(os.path.join(data_path, "*.pkl"))
    data_files_num = len(data_files)

    data_num = 0
    processed = 0
    total_mpjpe, total_pa_mpjpe, total_spa_mpjpe, total_error_R, total_error_t = [], [], [], [], []

    logger.debug("Start generating pseudo label dataset")
    for data_file in data_files:
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        pred_kpts3d, pred_score3d = [], []
        input_2d = data["input_2d"]
        for i in range(input_2d.shape[0]):
            kpts3d, score3d = estimate3d(cfg, model, input_2d[i])
            pred_kpts3d.append(kpts3d)
            pred_score3d.append(score3d)
        pred_kpts3d = np.array(pred_kpts3d)
        pred_score3d = np.array(pred_score3d)

        K = data["K"]
        pred_kpts2d = data["pred_kpts2d"]
        pred_score2d = data["pred_score2d"]
        pred_R, pred_t, tri_kpts3d = calibrate(cfg, pred_kpts2d, pred_score2d,
                                               pred_kpts3d, pred_score3d, K)

        gt_kpts3d = data["gt_kpts3d"]
        gt_pixel = data["gt_pixel"]
        gt_pixel_scale = data["gt_pixel_scale"]
        mpjpe, pa_mpjpe, spa_mpjpe = evaluate_plabels(tri_kpts3d, pred_R, pred_t, K,
                                                      gt_kpts3d, gt_pixel, gt_pixel_scale)

        gt_R = data["gtR"]
        gt_t = data["gtt"]
        error_R, error_t = evaluate_extrinsic(pred_R, pred_t, gt_R, gt_t)

        total_mpjpe.append(mpjpe)
        total_pa_mpjpe.append(pa_mpjpe)
        total_spa_mpjpe.append(spa_mpjpe)
        total_error_R.append(error_R)
        total_error_t.append(error_t)

        inputs_list, labels_list = [], []
        pred_kpts3d_cameras = to_pixel_coordinate(tri_kpts3d, pred_R, pred_t, K)
        for i in range(len(pred_kpts3d_cameras)):
            inputs, labels = split_data(input_2d[i], pred_kpts3d_cameras[i], None, 27, 27 // 3)
            inputs_list.append(inputs)
            labels_list.append(labels)

        inputs_list = np.array(inputs_list)
        labels_list = np.array(labels_list)
        for i in range(inputs_list.shape[0]):
            for j in range(inputs_list.shape[1]):
                data_input = inputs_list[i, j]
                data_label = labels_list[i, j]
                data_dict = {
                    'data_input': data_input,
                    'data_label': data_label,
                }
                save_path = os.path.join(train_path, f'{data_num:06d}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(data_dict, f)
                data_num += 1

        processed += 1
        if processed % 10 == 0:
            logger.debug(f"Processed {processed}/{data_files_num} files.")

    return (train_path, np.mean(total_mpjpe), np.mean(total_pa_mpjpe), np.mean(total_spa_mpjpe),
            np.mean(total_error_R), np.mean(total_error_t))
