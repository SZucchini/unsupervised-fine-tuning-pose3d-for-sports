import glob
import os
from logging import getLogger, StreamHandler, DEBUG, Formatter

import cv2
import numpy as np
import torch

from mmdet.apis import inference_detector
from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances
from natsort import natsorted

from . import _init_paths
from common.utils import turn_into_clips

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


h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]


def coco_h36m(kpts):
    temporal = kpts.shape[0]
    kpts_h36m = np.zeros_like(kpts, dtype=np.float32)
    htps_kpts = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_kpts: head, thorax, pelvis, spine
    htps_kpts[:, 0, 0] = np.mean(kpts[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_kpts[:, 0, 1] = np.sum(kpts[:, 1:3, 1], axis=1, dtype=np.float32) - kpts[:, 0, 1]
    htps_kpts[:, 1, :] = np.mean(kpts[:, 5:7, :], axis=1, dtype=np.float32)
    htps_kpts[:, 1, :] += (kpts[:, 0, :] - htps_kpts[:, 1, :]) / 3

    htps_kpts[:, 2, :] = np.mean(kpts[:, 11:13, :], axis=1, dtype=np.float32)
    htps_kpts[:, 3, :] = np.mean(kpts[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    kpts_h36m[:, spple_keypoints, :] = htps_kpts
    kpts_h36m[:, h36m_coco_order, :] = kpts[:, coco_order, :]

    kpts_h36m[:, 9, :] -= (kpts_h36m[:, 9, :]
                           - np.mean(kpts[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    kpts_h36m[:, 7, 0] += 2 * (kpts_h36m[:, 7, 0]
                               - np.mean(kpts_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    kpts_h36m[:, 8, 1] -= (np.mean(kpts[:, 1:3, 1], axis=1, dtype=np.float32) - kpts[:, 0, 1])*2/3

    valid_frames = np.where(np.sum(kpts_h36m.reshape(-1, 34), axis=1) != 0)[0]

    return kpts_h36m, valid_frames


def h36m_coco_format(keypoints, scores):
    h36m_kpts = []
    h36m_scores = []
    valid_frames = []
    new_score = np.zeros_like(scores, dtype=np.float32)

    if np.sum(keypoints) != 0.:
        kpts, valid_frame = coco_h36m(keypoints)
        h36m_kpts.append(kpts)
        valid_frames.append(valid_frame)

        new_score[:, h36m_coco_order] = scores[:, coco_order]
        new_score[:, 0] = np.mean(scores[:, [11, 12]], axis=1, dtype=np.float32)
        new_score[:, 8] = np.mean(scores[:, [5, 6]], axis=1, dtype=np.float32)
        new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
        new_score[:, 10] = np.mean(scores[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

        h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32)

    return h36m_kpts[0], h36m_scores[0], valid_frames


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2 or X.shape[-1] == 3
    result = np.copy(X)
    result[..., :2] = X[..., :2] / w * 2 - [1, h / w]
    return result


def process_one_image(cfg, img, detector, pose_estimator):
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == cfg.MMPOSE.DET_ID,
                                   pred_instance.scores > cfg.MMPOSE.DET_BOX_THR)]

    area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    bboxes = np.array([bboxes[np.argmax(area)]])

    bboxes = bboxes[nms(bboxes, cfg.MMPOSE.DET_NMS_THR), :4]
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    return data_samples.get('pred_instances', None)


def estimate2d(cfg, input_video, detector, pose_estimator):
    kpts2d = []
    score2d = []
    pred_instances_list = []
    cap = cv2.VideoCapture(input_video)
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if frame_idx == 0:
            img_size = frame.shape
        frame_idx += 1

        if not success:
            break

        pred_instances = process_one_image(cfg, frame, detector, pose_estimator)
        pred_instances_list = split_instances(pred_instances)

        kpt = np.array(pred_instances_list[0]['keypoints'])
        score = np.array(pred_instances_list[0]['keypoint_scores'])
        kpts2d.append(kpt)
        score2d.append(score)

    kpts2d = np.array(kpts2d)
    score2d = np.array(score2d)
    kpts2d, score2d, _ = h36m_coco_format(kpts2d, score2d)
    return kpts2d, score2d, img_size


def estimate3d(cfg, pose_lifter, kpts2d, score2d, img_size):
    pose_lifter.eval()
    kpts2d = kpts2d.reshape(1, *kpts2d.shape)
    score2d = score2d.reshape(1, *score2d.shape)
    keypoints = np.concatenate((kpts2d, score2d[..., None]), axis=-1)
    clips, downsample = turn_into_clips(keypoints, n_frames=cfg.FRAMES)

    kpts3d, score3d = [], []
    for idx, clip in enumerate(clips):
        input2d = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
        input2d = torch.from_numpy(input2d.astype('float32')).to(cfg.DEVICE)

        output_non_flip = pose_lifter(input2d)
        output_flip = pose_lifter(input2d)
        output = (output_non_flip + output_flip) / 2

        if idx == len(clips) - 1:
            output = output[:, downsample]
        output[:, :, 0, :] = 0
        post_out_all = output[0].cpu().detach().numpy()

        for post_out in post_out_all:
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value
            scores = np.ones((17), dtype='float32')

            kpts3d.append(post_out)
            score3d.append(scores)

    return np.array(kpts3d), np.array(score3d)
