from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from phasenet.conf.load_conf import TrainConfig
from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module,
                    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                    log: bool = False,
                    device=None) -> Optional[dict]:
    model.train()
    if log:
        loss_log = []
        predict_log = []
    for meta in data_loader:
        # * forward
        sgram, target = meta['sgram'].to(device), meta['label'].to(device)
        output = model(sgram)
        predict = output['predict']
        loss = criterion(predict, target)
        if log:
            loss_log.append(loss.item())
            predict_log.append(
                torch.nn.functional.softmax(predict.detach(), dim=1))
        # * backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    if log:
        return {
            "loss": loss_log,
            "loss_mean": np.mean(loss_log),
            "predict": predict_log
        }


def get_optimizer(params_to_optimize: Iterator[torch.nn.Parameter], train_conf: TrainConfig) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=train_conf.learning_rate, weight_decay=train_conf.weight_decay, amsgrad=False
    )
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, iters_per_epoch: int, train_conf: TrainConfig) -> torch.optim.lr_scheduler._LRScheduler:
    main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (iters_per_epoch * (train_conf.epochs -
                   train_conf.lr_warmup_epochs))) ** 0.9,
    )
    return main_lr_scheduler


def criterion(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.kl_div(
        torch.nn.functional.log_softmax(inputs, dim=1), target, reduction='batchmean',
    )
    return losses
