import argparse
import glob
import json
import os
import pickle

import c3d
import numpy as np
from natsort import natsorted

from libs.utils import Camera, get_kpts3d_pixel, normalize_kpts, split_data


def make_dir(path):
    """Creates a directory if it does not exist.

    Args:
        path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_camera(camera_file):
    """Loads camera parameters from a JSON file.

    Args:
        camera_file (str): Path to the JSON file.

    Returns:
        camera (Object): Camera object.
    """
    with open(camera_file, "r") as f:
        params = json.load(f)

    intrinsic = np.array(params['intrinsic_matrix']).reshape(3, 4)
    extrinsic = np.array(params['extrinsic_matrix']).reshape(4, 4)
    camera = Camera(intrinsic, extrinsic)
    return camera


def load_data2d(data2d_file):
    """Loads 2D data from a pickle file.

    Args:
        data2d_file (str): Path to the pickle file.

    Returns:
        pred_kpts2d (ndarray): Predicted 2D keypoints.
        pred_score2d (ndarray): Predicted 2D keypoint scores.
        img_size (tuple): Image size.
    """
    with open(data2d_file, 'rb') as f:
        data2d = pickle.load(f)

    pred_kpts2d = data2d['pred_kpts2d']  # (F, 17, 2)
    pred_score2d = data2d['pred_score2d']  # (F, 17)
    img_size = data2d['img_size']  # (H, W, C)
    return pred_kpts2d, pred_score2d, img_size


def load_gt_world_asp(gt_kpts_file):
    """Loads ground truth 3D keypoints from a C3D file and converts them to H36M format.

    Args:
        gt_kpts_file (str): Path to the C3D file.

    Returns:
        gt_kpts_world (ndarray): Ground truth 3D keypoints in H36M format.
    """
    gt_kpts_world = []
    reader = c3d.Reader(open(gt_kpts_file, 'rb'))
    for frame in reader.read_frames():
        kpt3d = frame[1][:, :3]
        gt_kpts_world.append(kpt3d)
    gt_kpts_world = np.array(gt_kpts_world)
    gt_kpts_world = asp_to_h36m(gt_kpts_world)
    return gt_kpts_world


def asp_to_h36m(kpts3d):
    """Converts keypoints from ASP to H36M format.

    Args:
        kpts3d (ndarray): 3D keypoints in ASP format.

    Returns:
        kpts3d_h36m (ndarray): 3D keypoints in H36M format.
    """
    kpts3d_h36m = np.zeros_like(kpts3d)
    h36m = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
    h36m_asp_order = [2, 1, 0, 8, 7, 6, 13, 12, 11, 10, 9, 5, 4, 3]
    kpts3d_h36m[:, h36m] = kpts3d[:, h36m_asp_order]
    kpts3d_h36m[:, 0] = (kpts3d[:, 2] + kpts3d[:, 8]) / 2
    kpts3d_h36m[:, 7] = (kpts3d[:, 2] + kpts3d[:, 8] + kpts3d[:, 5] + kpts3d[:, 11]) / 4
    kpts3d_h36m[:, 8] = (kpts3d[:, 5] + kpts3d[:, 11]) / 2
    kpts3d_h36m[:, 8] += (kpts3d[:, 13] - kpts3d_h36m[:, 8]) / 3
    return kpts3d_h36m


def process_data(camera_files, data2d_file, gt_kpts_file, frames=27):
    """Processes data for training.

    Args:
        camera_files (list): List of camera files.
        data2d_file (list): List of data2d files.
        gt_kpts_file (str): Path to the ground truth 3D keypoints file.
        frames (int): Number of frames to consider.

    Returns:
        camera_list (list): List of camera names.
        K_list (ndarray): List of intrinsic matrices.
        gtR_list (ndarray): List of rotation matrices.
        gtt_list (ndarray): List of translation vectors.
        pred_kpts2d_list (ndarray): List of predicted 2D keypoints.
        pred_score2d_list (ndarray): List of predicted 2D keypoint scores.
        input_2d_list (ndarray): List of input 2D data.
        gt_kpts3d_world (ndarray): Ground truth 3D keypoints in world coordinates.
        gt_pixel_list (ndarray): Ground truth 3D keypoints in pixel coordinates.
        gt_pixel_scale_list (ndarray): List of scales for ground truth 3D keypoints.
        inputs_list (ndarray): List of input data split into clips.
        labels_list (ndarray): List of label data split into clips.
        scales_list (ndarray): List of scales split into clips.
        denom_size_list (list): List of denormalization sizes.
    """
    dataset_name = gt_kpts_file.split('/')[2]
    data_split = gt_kpts_file.split('/')[3]
    if data_split == 'train':
        stride = 27 // 3
    else:
        stride = 27

    if dataset_name == 'ASPset-510':
        gt_kpts3d_world = load_gt_world_asp(gt_kpts_file)  # (F, 17, 3)
    elif dataset_name == 'Runner':
        gt_kpts3d_world_original = np.load(gt_kpts_file)
        gt_kpts3d_world = gt_kpts3d_world_original[::2] * 1000

    camera_list = [camera_file.split('/')[-1].split('.')[0] for camera_file in camera_files]
    K_list = np.array([load_camera(camera_file).K for camera_file in camera_files])
    gtR_list = np.array([load_camera(camera_file).R for camera_file in camera_files])
    gtt_list = np.array([load_camera(camera_file).t.reshape(3, 1) for camera_file in camera_files])

    pred_kpts2d_list, pred_score2d_list, input_2d_list = [], [], []
    inputs_list, labels_list, scales_list, denom_size_list = [], [], [], []
    gt_pixel_list, gt_pixel_scale_list = [], []
    for camera_file, data_2d_file in zip(camera_files, data2d_file):
        camera = load_camera(camera_file)

        pred_kpts2d, pred_score2d, img_size = load_data2d(data_2d_file)
        if dataset_name == 'Runner':
            pred_kpts2d = pred_kpts2d[::2]
            pred_score2d = pred_score2d[::2]
        input_kpts2d = normalize_kpts(pred_kpts2d, w=img_size[1], h=img_size[0])
        input_2d = np.concatenate([input_kpts2d, pred_score2d[..., None]], -1)

        gt_kpts3d_camera = camera.world_to_camera_space(gt_kpts3d_world)[:, :, :3]
        gt_kpts3d_pixel, scale_list = get_kpts3d_pixel(gt_kpts3d_camera, camera)
        label_3d = normalize_kpts(gt_kpts3d_pixel, w=img_size[1], h=img_size[0])

        inputs, labels, scales = split_data(input_2d, label_3d, scale_list, frames, stride)

        pred_kpts2d_list.append(pred_kpts2d)
        pred_score2d_list.append(pred_score2d)
        input_2d_list.append(input_2d)
        gt_pixel_list.append(gt_kpts3d_pixel)
        gt_pixel_scale_list.append(scale_list)

        inputs_list.append(inputs)
        labels_list.append(labels)
        scales_list.append(scales)
        denom_size_list.append(img_size)

    pred_kpts2d_list = np.array(pred_kpts2d_list)
    pred_score2d_list = np.array(pred_score2d_list)
    input_2d_list = np.array(input_2d_list)
    gt_pixel_list = np.array(gt_pixel_list)
    gt_pixel_scale_list = np.array(gt_pixel_scale_list)

    inputs_list = np.array(inputs_list)
    labels_list = np.array(labels_list)
    scales_list = np.array(scales_list)

    return (camera_list, K_list, gtR_list, gtt_list,
            pred_kpts2d_list, pred_score2d_list, input_2d_list,
            gt_kpts3d_world, gt_pixel_list, gt_pixel_scale_list,
            inputs_list, labels_list, scales_list, denom_size_list)


def main():
    """Preprocesses data for training.

    Args:
        data_root (str): Path to the data directory.
        frames (int): Number of frames per clip.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--frames", type=int, default=27)
    args = parser.parse_args()

    frames = args.frames
    data_root = args.data_root
    camera_root = os.path.join(data_root, 'cameras')
    dataset_name = data_root.split('/')[2]
    if dataset_name == 'ASPset-510':
        output_root = data_root.replace(dataset_name, f'ASP-{frames}')
    elif dataset_name == 'Runner':
        output_root = data_root.replace(dataset_name, f'R-{frames}')
    data_split = data_root.split('/')[-1]
    calib_root = os.path.join(os.path.dirname(output_root), f'calib_{data_split}')
    make_dir(output_root)
    make_dir(calib_root)

    data_2d_dirs = glob.glob(os.path.join(data_root, 'data_2d', '*'))
    if dataset_name == 'Runner':
        camera_files = natsorted(glob.glob(os.path.join(camera_root, '*.pkl')))

    data_num = 0
    for data_2d_dir in data_2d_dirs:
        data_name = data_2d_dir.split('/')[-1]  # 4d9e-0006
        gt_kpts_file = os.path.join(data_root, 'joints_3d',
                                    data_name.split('-')[0], f'{data_name}.c3d')

        data_2d_files = glob.glob(os.path.join(data_2d_dir, '*.pkl'))
        data_2d_files = natsorted(data_2d_files)

        if dataset_name == 'ASPset-510':
            camera_dir = os.path.join(camera_root, data_name.split('-')[0])
            camera_files = natsorted(glob.glob(os.path.join(camera_dir, '*.json')))

        (camera_list, K_list, gtR_list, gtt_list,
         pred_kpts2d_list, pred_score2d_list, input_2d_list,
         gt_kpts3d_world, gt_pixel_list, gt_pixel_scale_list,
         inputs_list, labels_list, scales_list, denom_size_list) = process_data(camera_files,
                                                                                data_2d_files,
                                                                                gt_kpts_file,
                                                                                frames)

        for i in range(inputs_list.shape[0]):
            for j in range(inputs_list.shape[1]):
                data_input = inputs_list[i, j]
                data_label = labels_list[i, j]
                data_scale = scales_list[i, j]
                denom_size = denom_size_list[i]
                data_dict = {
                    'data_input': data_input,
                    'data_label': data_label,
                    'data_scale': data_scale,
                    'denom_size': denom_size
                }
                save_path = os.path.join(output_root, f'{data_num:06d}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(data_dict, f)
                data_num += 1

        if data_split != "test":
            calib_dict = {
                'camera_list': camera_list,
                'K': K_list,
                'gtR': gtR_list,
                'gtt': gtt_list,
                'gt_kpts3d': gt_kpts3d_world,
                'gt_pixel': gt_pixel_list,
                'gt_pixel_scale': gt_pixel_scale_list,
                'pred_kpts2d': pred_kpts2d_list,
                'pred_score2d': pred_score2d_list,
                'input_2d': input_2d_list,
            }
            save_path = os.path.join(calib_root, f'{data_name}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(calib_dict, f)


if __name__ == '__main__':
    main()
