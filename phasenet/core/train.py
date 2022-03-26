

import datetime
import os
import time

import torch
import torch.nn as nn
import torch.utils.data
import torchvision

import phasenet.models
# from phasenet.models.unet import UNet
import utils
from phasenet.utils.data_reader import DataReader_test, DataReader_train
from phasenet.utils.visulization import plot_spectrogram, plot_waveform


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        # losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
        losses[name] = nn.functional.kl_div(
            torch.nn.functional.log_softmax(x, dim=1), target, reduction='mean',
        )

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes=3, epoch=0):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.inference_mode():
        for meta in metric_logger.log_every(data_loader, 100, header):
            data = meta["data"]
            target = meta["target"]
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            output = output["out"]

            metric_logger.update(loss=loss.item())
            confmat.update(target.argmax(1).flatten(),
                           output.argmax(1).flatten())

        if utils.is_main_process():
            plot_waveform(
                data.permute(0, 2, 3, 1).cpu().numpy(),
                torch.nn.functional.softmax(output, dim=1)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy(),
                label=target.permute(0, 2, 3, 1).cpu().numpy(),
                epoch=epoch,
                dt=1.0 / 100.0,
            )
        # if "sgram" in result:
        #     spectrogram = result["sgram"]
        #     plot_spectrogram(spectrogram.permute(0,2,3,1).cpu().numpy(), torch.nn.functional.softmax(output, dim=1).permute(0,2,3,1).cpu().numpy(), label=target.permute(0,2,3,1).cpu().numpy(), epoch=epoch, dt=8/40)
        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    lr_scheduler,
    device,
    epoch,
    print_freq,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for meta in metric_logger.log_every(data_loader, print_freq, header):
        data = meta["data"]
        target = meta["target"]
        data, target = data.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(data)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
