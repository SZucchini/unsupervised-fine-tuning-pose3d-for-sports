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

import torch
from torch import optim
from torch.utils.data import DataLoader

from .dataset import MotionDataset3D
from .loss import loss_mpjpe, n_mpjpe, loss_velocity


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


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    loss_epoch = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y = y - y[..., 0:1, :]

        pred = model(x)
        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)

        loss_total = loss_3d_pos + loss_3d_scale * 0.5 + loss_3d_velocity * 20
        loss_total.backward()
        optimizer.step()

        loss_epoch += loss_total.item()

    loss_epoch /= len(train_loader)
    return loss_epoch


def valid_one_epoch(model, test_loader, device):
    model.eval()
    mpjpe = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y = y - y[..., 0:1, :]

        pred = model(x)
        mpjpe += loss_mpjpe(pred, y).item()

    mpjpe /= len(test_loader)
    return mpjpe


def train(cfg, run, model, train_path, valid_path, itr):
    workspace = cfg.WORKSPACE
    best_path = f"{workspace}/checkpoint/iter{itr}_best.pth.tr"
    last_path = f"{workspace}/checkpoint/iter{itr}_last.pth.tr"

    train_dataset = MotionDataset3D(train_path)
    valid_dataset = MotionDataset3D(valid_path)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=cfg.TUNING.BATCHSIZE, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False,
                              batch_size=cfg.TUNING.BATCHSIZE, pin_memory=True)

    lr = cfg.TUNING.LEARNING_RATE
    lr_decay = cfg.TUNING.LR_DECAY
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=cfg.TUNING.WEIGHT_DECAY)

    epoch_start = 0
    min_mpjpe = float('inf')
    for epoch in range(epoch_start, cfg.TUNING.EPOCHS):
        train_epoch_loss = train_one_epoch(model, train_loader, optimizer, cfg.DEVICE)
        valid_epoch_mpjpe = valid_one_epoch(model, valid_loader, cfg.DEVICE)

        if run is not None:
            run[f"iter{itr}/train/epoch_loss"].log(train_epoch_loss)
            run[f"iter{itr}/valid/epoch_mpjpe"].log(valid_epoch_mpjpe)

        if valid_epoch_mpjpe < min_mpjpe:
            min_mpjpe = valid_epoch_mpjpe
            best_model = copy.deepcopy(model)
            save_checkpoint(best_path, epoch, lr, optimizer, model, min_mpjpe)

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)
        if epoch == cfg.TUNING.EPOCHS - 1:
            save_checkpoint(last_path, epoch, lr, optimizer, model, min_mpjpe)

    return best_model
