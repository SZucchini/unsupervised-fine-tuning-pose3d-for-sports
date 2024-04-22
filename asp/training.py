import copy
from logging import getLogger, StreamHandler, DEBUG, Formatter

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from .dataset import flip_data, MotionDataset3D
from .loss import loss_mpjpe, n_mpjpe, loss_velocity, calc_mpjpe, p_mpjpe

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


def train_one_epoch(cfg, run, model, train_loader, optimizer, itr=0):
    model.train()
    loss_epoch = 0
    for x, y in train_loader:
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)

        with torch.no_grad():
            y = y - y[..., 0:1, :]

        pred = model(x)
        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)

        loss_total = loss_3d_pos \
            + loss_3d_scale * cfg.TRAIN.SCALE_WEIGHT \
            + loss_3d_velocity * cfg.TRAIN.VELOCITY_WEIGHT
        loss_total.backward()
        optimizer.step()

        run[f"train/{itr}/batch/loss_3d_pos"].log(loss_3d_pos.item())
        run[f"train/{itr}/batch/loss_3d_scale"].log(loss_3d_scale.item())
        run[f"train/{itr}/batch/loss_3d_velocity"].log(loss_3d_velocity.item())
        run[f"train/{itr}/batch/loss_total"].log(loss_total.item())

        loss_epoch += loss_total.item()

    loss_epoch /= len(train_loader)
    return loss_epoch


def denormalize(kpts, denom_w, denom_h):
    """
    kpts    : np.darray [B, N, 17, 3]
    denom_w : np.darray [B]
    denom_h : np.darray [B]
    """
    for idx, _ in enumerate(kpts):
        kpts[idx, :, :, :2] = ((kpts[idx, :, :, :2] +
                                np.array([1, denom_h[idx] / denom_w[idx]])) * denom_w[idx] / 2)
        kpts[idx, :, :, 2:] = kpts[idx, :, :, 2:] * denom_w[idx] / 2
    return kpts


def test_one_epoch(cfg, model, test_loader):
    model.eval()

    mpjpe_list = []
    pa_mpjpe_list = []
    with torch.no_grad():
        for x, y, scale, denom in test_loader:
            x = x.to(cfg.DEVICE)

            batch_input_flip = flip_data(x)
            pred_1 = model(x)
            pred_flip = model(batch_input_flip)
            pred_2 = flip_data(pred_flip)
            pred = (pred_1 + pred_2) / 2

            pred[:, :, 0, :] = 0

            pred = pred.cpu().numpy()
            y = y.cpu().numpy()

            scale = scale.cpu().numpy()
            denom_w = denom[1].cpu().numpy()
            denom_h = denom[0].cpu().numpy()

            pred_denom = denormalize(pred, denom_w, denom_h)
            y_denom = denormalize(y, denom_w, denom_h)
            pred_denom = pred_denom - pred_denom[:, :, 0:1, :]
            y_denom = y_denom - y_denom[:, :, 0:1, :]

            pred_denom = pred_denom / scale[:, :, None, None]
            y_denom = y_denom / scale[:, :, None, None]

            mpjpe = calc_mpjpe(pred_denom, y_denom)  # [B, N, 17, 3]
            mpjpe = np.mean(mpjpe)
            mpjpe_list.append(mpjpe)

            for i in range(pred_denom.shape[0]):
                pa_mpjpe = p_mpjpe(pred_denom[i], y_denom[i])
                pa_mpjpe_list.append(pa_mpjpe)

    return np.mean(mpjpe_list), np.mean(pa_mpjpe_list)


def decay_lr_exponentially(lr, lr_decay, optimizer):
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return lr


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
    }, checkpoint_path)


def train(cfg, run, model, train_path, itr=0):
    workspace = cfg.WORKSPACE
    best_path = f"{workspace}/checkpoint/iter{itr}_best.pth.tr"
    last_path = f"{workspace}/checkpoint/iter{itr}_last.pth.tr"

    train_path = train_path
    test_path = cfg.DATA.TEST_PATH

    train_dataset = MotionDataset3D(cfg, train_path)
    test_dataset = MotionDataset3D(cfg, test_path, test=True)
    logger.debug(f"Train dataset size: {len(train_dataset)}")
    logger.debug(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=cfg.TRAIN.BATCHSIZE, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=cfg.TRAIN.BATCHSIZE, pin_memory=True)

    lr = cfg.TRAIN.LEARNING_RATE
    lr_decay = cfg.TRAIN.LR_DECAY
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    test_mpjpe, test_pa_mpjpe = test_one_epoch(cfg, model, test_loader)
    logger.debug("Initial Evaluation.")
    logger.debug(f"Test MPJPE: {test_mpjpe} [mm]")
    logger.debug(f"Test PA-MPJPE: {test_pa_mpjpe} [mm]")

    epoch_start = 0
    min_pa_mpjpe = float('inf')
    for epoch in range(epoch_start, cfg.TRAIN.EPOCHS):
        train_epoch_loss = train_one_epoch(cfg, run, model, train_loader, optimizer, itr=itr)
        test_epoch_mpjpe, test_epoch_pa_mpjpe = test_one_epoch(cfg, model, test_loader)

        run[f"train/{itr}/epoch/loss"].log(train_epoch_loss)
        run[f"test/{itr}/epoch/mpjpe"].log(test_epoch_mpjpe)
        run[f"test/{itr}/epoch/pa_mpjpe"].log(test_epoch_pa_mpjpe)
        logger.debug(f"Epoch {epoch}")
        logger.debug(f"Test MPJPE: {test_epoch_mpjpe} [mm]")
        logger.debug(f"Test PA-MPJPE: {test_epoch_pa_mpjpe} [mm]")

        if test_epoch_pa_mpjpe < min_pa_mpjpe:
            min_pa_mpjpe = test_epoch_pa_mpjpe
            best_model = copy.deepcopy(model)
            save_checkpoint(best_path, epoch, lr, optimizer, best_model, min_pa_mpjpe)

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)

        if epoch == cfg.TRAIN.EPOCHS - 1:
            save_checkpoint(last_path, epoch, lr, optimizer, model, min_pa_mpjpe)
            test_mpjpe, test_pa_mpjpe = test_one_epoch(cfg, model, test_loader)
            test_mpjpe_best, test_pa_mpjpe_best = test_one_epoch(cfg, best_model, test_loader)

            run["test/best_model/mpjpe"].log(test_mpjpe_best)
            run["test/best_model/pa_mpjpe"].log(test_pa_mpjpe_best)
            logger.debug(f"Test MPJPE Best model: {test_mpjpe_best} [mm]")
            logger.debug(f"Test PA-MPJPE Best model: {test_pa_mpjpe_best} [mm]")

    return best_model
