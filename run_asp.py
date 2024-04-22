import argparse
import os
import random
from logging import getLogger, StreamHandler, DEBUG, Formatter

import neptune
import numpy as np
import torch
import torch.nn as nn

from asp.databuild import generate_plabel_dataset
from asp.training import train
from common.MotionAGFormer.model import MotionAGFormer
from config.asp.defaults import get_cfg_defaults

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
    work_dirs = ["/checkpoint", "/dataset"]
    for work_dir in work_dirs:
        path = cfg.WORKSPACE + work_dir
        if not os.path.exists(path):
            os.makedirs(path)


def load_model(cfg):
    """Load model.

    Args:
        cfg (CfgNode): Configuration.

    Returns:
        model (nn.Module): Model.
    """
    model_path = cfg.PRETRAINED_CKPT
    model = MotionAGFormer(n_layers=12, dim_in=3, dim_feat=64,
                           num_heads=8, neighbour_num=2, n_frames=27)
    model = nn.DataParallel(model)
    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.to(cfg.DEVICE)
    return model


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
    parser.add_argument("--workspace", type=str, default="./workspace/default")
    parser.add_argument("--neptune", action="store_true")
    parser.add_argument("--neptune_project", type=str, default="username/project")
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

    logger.debug("Loading model ...")
    model = load_model(cfg)

    iterations = cfg.TUNING.ITERATIONS
    for itr in range(1, iterations+1):
        logger.debug(f"Start training iteration {itr} ...")
        if iterations == 1:
            train_path = cfg.DATA.TRAIN_PATH
            logger.debug("Train model using Ground Truth Labels.")
        else:
            logger.debug("Train model using Pseudo Labels.")
            logger.debug("Generate Pseudo Labels.")
            (train_path, mpjpe, pa_mpjpe, spa_mpjpe,
                R_errors, t_errors) = generate_plabel_dataset(cfg, model, itr)
            logger.debug("Psuedo Labels Evaluation.")
            logger.debug(f"MPJPE: {mpjpe} mm, PA_MPJPE: {pa_mpjpe} mm, SPA_MPJPE: {spa_mpjpe} mm")
            logger.debug(f"R_errors: {R_errors}, t_errors: {t_errors}")

        model = train(cfg, run, model, train_path, itr)
        logger.debug(f"Training iteration {itr} finished.")

    run.stop()


if __name__ == "__main__":
    main()
