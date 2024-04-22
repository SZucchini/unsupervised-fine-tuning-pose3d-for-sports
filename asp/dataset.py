import copy
import glob
import pickle
import random

import torch
from natsort import natsorted
from torch.utils.data import Dataset


def read_pkl(data_url):
    file = open(data_url, 'rb')
    content = pickle.load(file)
    file.close()
    return content


def flip_data(data, left=[1, 2, 3, 14, 15, 16], right=[4, 5, 6, 11, 12, 13]):
    """
    data: [B, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left + right, :] = flipped_data[..., right + left, :]
    return flipped_data


class MotionDataset3D(Dataset):
    def __init__(self, cfg, data_root, test=False, flip=True):
        self.data_root = data_root
        self.data_split = data_root.split('/')[-1]
        if ('train' in self.data_split) and cfg.TRAIN.WITH_H36M:
            self.data_root = [self.data_root, cfg.DATA.H36M_PATH]
            self.h36m_ratio = 1
        self.flip = flip
        self.test = test
        self.file_list = self._generate_file_list()

    def _generate_file_list(self):
        if isinstance(self.data_root, list):
            file_list = []
            for i, data_root in enumerate(self.data_root):
                files = natsorted(glob.glob(data_root + '/*.pkl'))
                if i == 0:
                    file_num = len(files)
                if i == 1:
                    file_num = int(file_num * self.h36m_ratio)
                    files = random.sample(files, file_num)
                file_list += files
        else:
            files = glob.glob(self.data_root + '/*.pkl')
            file_list = natsorted(files)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        if 'train' in self.data_split and not self.test:
            motion_2d = motion_file["data_input"]
            motion_3d = motion_file["data_label"]
            if self.flip and random.random() > 0.5:
                motion_2d = flip_data(motion_2d)
                motion_3d = flip_data(motion_3d)
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)

        else:
            motion_2d = motion_file["data_input"]
            motion_3d = motion_file["data_label"]
            scale = motion_file["data_scale"]
            denom = motion_file["denom_size"]
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), scale, denom
