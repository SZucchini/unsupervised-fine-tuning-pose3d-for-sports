"""
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
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left + right, :] = flipped_data[..., right + left, :]
    return flipped_data


class MotionDataset3D(Dataset):
    def __init__(self, data_root, flip=True):
        self.data_root = data_root
        self.data_split = data_root.split('/')[-1]
        self.flip = flip
        self.file_list = self._generate_file_list()

    def _generate_file_list(self):
        files = glob.glob(self.data_root + '/*.pkl')
        file_list = natsorted(files)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        motion_2d = motion_file["data_input"]
        motion_3d = motion_file["data_label"]

        if self.data_split == 'train':
            if self.flip and random.random() > 0.5:
                motion_2d = flip_data(motion_2d)
                motion_3d = flip_data(motion_3d)

        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)
