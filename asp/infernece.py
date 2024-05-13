import numpy as np
import torch

from common.utils import turn_into_clips
from .dataset import flip_data


def estimate3d(cfg, model, input_2d):
    """
    input_2d : np.ndarray, (F, 17, 3)
    """
    input_2d = input_2d.reshape(1, *input_2d.shape)
    clips, downsample = turn_into_clips(input_2d, n_frames=cfg.FRAMES)

    kpts3d = []
    model.eval()
    for idx, clip in enumerate(clips):
        with torch.no_grad():
            inputs = torch.from_numpy(clip.astype(np.float32)).to(cfg.DEVICE)
            inputs_flip = flip_data(inputs)
            output_1 = model(inputs)
            output_flip = model(inputs_flip)
            output_2 = flip_data(output_flip)
            output = (output_1 + output_2) / 2

        if idx == len(clips) - 1:
            output = output[:, downsample]

        output[:, :, 0, :] = 0
        post_out_all = output[0].cpu().detach().numpy()
        for post_out in post_out_all:
            kpts3d.append(post_out)

    kpts3d = np.array(kpts3d)
    socre3d = np.ones_like(kpts3d[:, :, 0])

    return kpts3d, socre3d
