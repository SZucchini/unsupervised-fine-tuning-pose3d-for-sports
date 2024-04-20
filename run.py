import argparse
import os
import pickle
import random
from logging import getLogger, StreamHandler, DEBUG, Formatter

import neptune
import numpy as np
import torch
import torch.nn as nn
from mmdet.apis import init_detector
from mmpose.apis import init_model
from mmpose.utils import adapt_mmdet_pipeline

from config.runner.defaults import get_cfg_defaults
from common.MotionAGFormer.model import MotionAGFormer
# from libs.databuilder import generate_dataset, get_data2d, get_data_dict, get_gt_dict
# from libs.eval.evaluation import eval_calib, eval_mono
# from libs.pose.MotionAGFormer.model import MotionAGFormer
# from libs.tuning.training import train

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


def set_seed(seed):
    """Set seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_cfg(config, opts):
    """Get configuration.

    Args:
        config (str): Configuration file.
        opts (list): List of options.

    Returns:
        cfg (CfgNode): Configuration.
    """
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def init_neptune(cfg):
    """Initialize Neptune.

    Args:
        cfg (CfgNode): Configuration.

    Returns:
        run (neptune.run.Run): Neptune run.
    """
    with open(cfg.NEPTUNE.TOKEN_PATH, "r") as f:
        api_token = f.readline().rstrip("\n")
    run = neptune.init_run(
        project=cfg.NEPTUNE.PROJECT,
        api_token=api_token,
    )
    return run


def upload_cfg(cfg, run):
    """Upload configuration to Neptune.

    Args:
        cfg (CfgNode): Configuration.
        run (neptune.run.Run): Neptune run.
    """
    cfg_path = cfg.WORKSPACE + '/config.yaml'
    with open(cfg_path, 'w') as f:
        f.write(cfg.dump())
    if run is not None:
        run["config"].upload(cfg_path)


def create_workspace(cfg):
    """Create workspace directories.

    Args:
        cfg (CfgNode): Configuration.
    """
    if not os.path.exists(cfg.WORKSPACE):
        os.makedirs(cfg.WORKSPACE)
    work_dirs = ["/dataset", "/evaluation", "/checkpoint"]
    for work_dir in work_dirs:
        path = cfg.WORKSPACE + work_dir
        if not os.path.exists(path):
            os.makedirs(path)


def load_models(cfg):
    """Load models.

    Args:
        cfg (CfgNode): Configuration.

    Returns:
        detector (nn.Module): Detector model.
        pose_estimator (nn.Module): Pose estimator model (2D Pose estimation).
        pose_lifter (nn.Module): Pose lifter model (3D Pose estimation).
    """
    det_config = cfg.MMPOSE.DET_CONFIG
    det_ckpt = cfg.MMPOSE.DET_CKPT
    detector = init_detector(det_config, det_ckpt, device=cfg.DEVICE)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_config = cfg.MMPOSE.VITPOSE_CONFIG
    pose_ckpt = cfg.MMPOSE.VITPOSE_CKPT
    pose_estimator = init_model(pose_config, pose_ckpt, device=cfg.DEVICE)

    model_path = cfg.MOTIONAGF_CKPT
    if model_path.split('/')[-1].split('-')[1] == 'l':
        pose_lifter = MotionAGFormer(n_layers=26, dim_in=3, dim_feat=128,
                                     num_heads=8, neighbour_num=2)
    elif model_path.split('/')[-1].split('-')[1] == 'b':
        pose_lifter = MotionAGFormer(n_layers=16, dim_in=3, dim_feat=128,
                                     num_heads=8, neighbour_num=2)
    elif model_path.split('/')[-1].split('-')[1] == 'xs':
        pose_lifter = MotionAGFormer(n_layers=12, dim_in=3, dim_feat=64,
                                     num_heads=8, neighbour_num=2, n_frames=27)
    pose_lifter = nn.DataParallel(pose_lifter)
    pre_dict = torch.load(model_path)
    pose_lifter.load_state_dict(pre_dict['model'], strict=True)
    pose_lifter.eval()
    pose_lifter.to(cfg.DEVICE)
    return detector, pose_estimator, pose_lifter


def main():
    """Main function.

    Args:
        config (str): Configuration file path.
        workspace (str): Workspace directory.
        neptune (bool): Neptune flag.
        neptune_project (str): Neptune project.
        neptune_token (str): Neptune token path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/MotionAGFormer.yaml")
    parser.add_argument("--workspace", type=str, default="./data/run_001/results")
    parser.add_argument("--neptune", action="store_true")
    parser.add_argument("--neptune_project", type=int, default="username/project")
    parser.add_argument("--neptune_token", type=str, default="./token/neptune.txt")
    args = parser.parse_args()

    opts = [
        "WORKSPACE", args.workspace,
        "NEPTUNE.PROJECT", args.neptune_project,
        "NEPTUNE.TOKEN_PATH", args.neptune_token
    ]
    cfg = get_cfg(args.config, opts)
    logger.debug("Config data\n{}\n".format(cfg))

    if args.neptune:
        run = init_neptune(cfg)
    else:
        run = None
    upload_cfg(cfg, run)
    set_seed(cfg.SEED)
    create_workspace(cfg)

    logger.debug("Loading models ...")
    detector, pose_estimator, pose_lifter = load_models(cfg)

    logger.debug("Loading ground truth data for evaluation ...")
    gt_dict = get_gt_dict(cfg)
    with open(cfg.WORKSPACE + '/evaluation/gt_dict.pkl', 'wb') as f:
        pickle.dump(gt_dict, f)

    logger.debug("Building initial data ...")
    dataset_dict, eval_dict = get_data_dict(
        cfg,
        pose_lifter,
        get_data2d(cfg, detector, pose_estimator)
    )
    with open(cfg.WORKSPACE + '/evaluation/init_dict.pkl', 'wb') as f:
        pickle.dump(eval_dict, f)
    logger.debug("Finished.")

    logger.debug("Initial Evaluation")
    eval_calib(gt_dict, eval_dict)
    eval_mono(gt_dict, eval_dict)

    for itr in range(cfg.TUNING.ITERATIONS):
        logger.debug("Iteration %s start ...", itr+1)
        logger.debug("Generating dataset ...")
        train_path, valid_path = generate_dataset(cfg, dataset_dict, itr+1)
        logger.debug("Train dataset path: %s", train_path)
        logger.debug("Valid dataset path: %s", valid_path)

        logger.debug("Start training ...")
        pose_lifter = train(cfg, run, pose_lifter, train_path, valid_path, itr+1)
        logger.debug("Training finished.")

        logger.debug("Building new data ...")
        dataset_dict, eval_dict = get_data_dict(
            cfg,
            pose_lifter,
            get_data2d(cfg, detector, pose_estimator)
        )
        logger.debug("Finished.")
        with open(cfg.WORKSPACE + f'/evaluation/iter{itr+1}.pkl', 'wb') as f:
            pickle.dump(eval_dict, f)

        logger.debug("Iteration {} Evaluation".format(itr+1))
        eval_calib(gt_dict, eval_dict)
        eval_mono(gt_dict, eval_dict)
        logger.debug("Iteration %s finished.", itr+1)

    if run is not None:
        run.stop()


if __name__ == "__main__":
    main()
