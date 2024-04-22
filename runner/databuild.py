import glob
import os
import pickle

import numpy as np
from natsort import natsorted
from pycalib.calib import rebase_all

from . import _init_paths
from common.calibration import calibrate
from common.utils import invRT_batch, normalize_kpts
from .inference import estimate2d, estimate3d
from .utils import get_w2c_params, split_clips, world_to_camera


def get_gt_dict(cfg):
    gt_dict = {}
    for data_name in cfg.TUNING.EVAL_DATA:
        gt_dict[data_name] = []
        gt_path = f"./data/Runner/{data_name}/gt_kpts"
        gt_files = natsorted(glob.glob(gt_path + '/*.npy'))

        for gt_file in gt_files:
            gt_kpts = np.load(gt_file)
            gt_dict[data_name].append(gt_kpts)

    return gt_dict


def get_data2d(cfg, detector, pose_estimator):
    if os.path.exists(cfg.TUNING.DATA2D_PATH):
        with open(cfg.TUNING.DATA2D_PATH, 'rb') as f:
            data2d = pickle.load(f)

    else:
        data2d = {}
        data_dirs = cfg.TUNING.DATA_DIRS

        for data_dir in data_dirs:
            data_name = data_dir.split('/')[-2]
            input_dirs = natsorted(glob.glob(data_dir + '/*'))
            video_dirs = [natsorted(glob.glob(input_dir + '/*.mp4')) for input_dir in input_dirs]

            data2d[data_name] = {}
            data2d[data_name]['kpts'] = []
            data2d[data_name]['score'] = []
            data2d[data_name]['img_size'] = []

            for video_files in video_dirs:
                k2d_set, s2d_set, size_set = [], [], []
                for input_video in video_files:
                    k2d, s2d, img_size = estimate2d(cfg, input_video, detector, pose_estimator)
                    k2d_set.append(k2d)
                    s2d_set.append(s2d)
                    size_set.append(img_size)

                data2d[data_name]['kpts'].append(np.array(k2d_set))  # (C, F, 17, 2)
                data2d[data_name]['score'].append(np.array(s2d_set))  # (C, F, 17)
                data2d[data_name]['img_size'].append(size_set)  # (C, 2)

        with open(cfg.TUNING.DATA2D_PATH, 'wb') as f:
            pickle.dump(data2d, f)

    return data2d


def get_data_dict(cfg, pose_lifter, data2d):
    data3d = {}
    dataset_dict, eval_dict = {}, {}
    param_c1 = np.load(cfg.CALIB.INTRINSIC_C1)
    param_c2 = np.load(cfg.CALIB.INTRINSIC_C2)
    param_c3 = np.load(cfg.CALIB.INTRINSIC_C3)
    K = np.array([param_c1["mtx"], param_c2["mtx"], param_c3["mtx"]])

    for data_name in data2d.keys():
        data3d[data_name] = {}
        data3d[data_name]['kpts'] = []
        data3d[data_name]['score'] = []

        for k2d_set, s2d_set, size_set in zip(data2d[data_name]['kpts'],
                                              data2d[data_name]['score'],
                                              data2d[data_name]['img_size']):
            k3d_set, s3d_set = [], []
            for k2d, s2d, img_size in zip(k2d_set, s2d_set, size_set):
                k3d, s3d = estimate3d(cfg, pose_lifter, k2d, s2d, img_size)
                k3d_set.append(k3d)
                s3d_set.append(s3d)

            data3d[data_name]['kpts'].append(np.array(k3d_set))
            data3d[data_name]['score'].append(np.array(s3d_set))

    for data_name in data3d.keys():
        if data_name in cfg.TUNING.TRAIN_DATA:
            dataset_dict[data_name] = {}
            dataset_dict[data_name]['3d_kpts_triangulated'] = []
            dataset_dict[data_name]['R_est'] = []
            dataset_dict[data_name]['t_est'] = []
            dataset_dict[data_name]['2d_kpts'] = data2d[data_name]['kpts']
            dataset_dict[data_name]['2d_scores'] = data2d[data_name]['score']
            dataset_dict[data_name]['img_size'] = data2d[data_name]['img_size']

        if data_name in cfg.TUNING.EVAL_DATA:
            eval_dict[data_name] = {}
            eval_dict[data_name]['3d_kpts_triangulated'] = []
            eval_dict[data_name]['3d_kpts_by_cam'] = data3d[data_name]['kpts']

        kpts2d = data2d[data_name]['kpts']
        score2d = data2d[data_name]['score']
        kpts3d = data3d[data_name]['kpts']
        score3d = data3d[data_name]['score']

        for k2d, s2d, k3d, s3d in zip(kpts2d, score2d, kpts3d, score3d):
            R_est, t_est, k3d_tri = calibrate(
                cfg, k2d, s2d, k3d, s3d, K
            )

            if data_name in cfg.TUNING.TRAIN_DATA:
                dataset_dict[data_name]['3d_kpts_triangulated'].append(k3d_tri)
                dataset_dict[data_name]['R_est'].append(R_est)
                dataset_dict[data_name]['t_est'].append(t_est)

            if data_name in cfg.TUNING.EVAL_DATA:
                eval_dict[data_name]['3d_kpts_triangulated'].append(k3d_tri)

    return dataset_dict, eval_dict


def generate_labelset(k2d, s2d, k3d, size, R, t):
    R_new, t_new = rebase_all(R, t, normalize_scaling=True)
    R_c2w, t_c2w = invRT_batch(R_new, t_new)
    orientations, translations = get_w2c_params(R_c2w, t_c2w)

    k3d = k3d.astype(np.float32)
    gt_kpts = []
    for i in range(len(orientations)):
        pos3d = world_to_camera(k3d, orientations[i], translations[i])
        gt_kpts.append(pos3d)
    gt_kpts = np.array(gt_kpts)

    k2d_norm = np.zeros_like(k2d)
    for i in range(len(k2d)):
        k2d_norm[i] = normalize_kpts(k2d[i], size[i][1], size[i][0])  # TODO: modified
    s2d_reshaped = s2d[..., np.newaxis]
    input2d = np.concatenate([k2d_norm, s2d_reshaped], axis=-1)

    split_idx = split_clips(gt_kpts.shape[1], 27, 27 // 3)
    inputs = input2d[:, split_idx, :]
    labels = gt_kpts[:, split_idx, :]

    return inputs, labels


def generate_dataset(cfg, dataset_dict, itr):
    dataset_path = cfg.WORKSPACE + f'/dataset/iter{itr}'
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    for data_name in dataset_dict.keys():
        kpts2d = dataset_dict[data_name]['2d_kpts']
        score2d = dataset_dict[data_name]['2d_scores']
        R_est = dataset_dict[data_name]['R_est']
        t_est = dataset_dict[data_name]['t_est']
        kpts3d = dataset_dict[data_name]['3d_kpts_triangulated']
        img_size = dataset_dict[data_name]['img_size']

        train_cnt, valid_cnt = 0, 0
        train_idx = int(len(kpts2d) * cfg.TUNING.TRAIN_RATIO)
        for i, (k2d, s2d, k3d, size, R, t) in enumerate(zip(kpts2d, score2d, kpts3d,
                                                            img_size, R_est, t_est)):
            inputs, labels = generate_labelset(k2d, s2d, k3d, size, R, t)

            for j in range(labels.shape[0]):
                for k in range(labels.shape[1]):
                    data_input = inputs[j, k]
                    data_label = labels[j, k]
                    data_dict = {
                        'data_input': data_input,
                        'data_label': data_label,
                    }

                    if i < train_idx:
                        train_cnt += 1
                        save_path = os.path.join(train_path, f'{data_name}_{train_cnt:08d}.pkl')
                        with open(save_path, 'wb') as f:
                            pickle.dump(data_dict, f)
                    else:
                        valid_cnt += 1
                        save_path = os.path.join(valid_path, f'{data_name}_{valid_cnt:08d}.pkl')
                        with open(save_path, 'wb') as f:
                            pickle.dump(data_dict, f)

    return train_path, valid_path
