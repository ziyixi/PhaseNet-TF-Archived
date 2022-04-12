"""
visualize.py

helper functions to visualzie the dataset and model.
"""
import os
from os.path import join
from typing import List, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import cm
from phasenet.conf.load_conf import Config
from torch.utils.data import DataLoader
from tqdm import tqdm


class BatchInput(TypedDict):
    data: torch.Tensor
    arrivals: torch.Tensor
    key: List[str]
    sgram: torch.Tensor
    label: torch.Tensor


def show_info(input_batch: BatchInput, phases: List[str], save_dir: str, sampling_rate: int, x_range: List[int], freq_range: List[int], merge: bool = False, global_max: bool = False, cur_example_num: int = 0, progress: bool = False, predict: Optional[torch.Tensor] = None) -> None:
    """show input dataset and save to pdf files

    Args:
        data_batch (BatchInput): batched dataset
        phases (List[str]): seismic phases name
        save_dir (str): the saving directory
        sampling_rate (int): sampling rate for the events
        x_range (List[int]): the x axis time range
        freq_range (List[int]): the freq axis range
        merge (bool): if merge to a single pdf file, the name will be input.pdf
        global_max (bool): for sgram, if use the same vmax value for three components
        cur_example_num (int): how many pdfs to generate in total
        progress (bool): if show the progress bar
    """
    data_batch, arrivals_batch, key_batch, sgram_batch, label_batch = input_batch[
        'data'].detach(), input_batch["arrivals"].detach(), input_batch["key"], input_batch["sgram"].detach(), input_batch["label"].detach()
    sgram_batch = sgram_batch.cpu()
    if predict != None:
        # show predict instead
        label_batch = predict.cpu()
    batch_size = data_batch.shape[0]
    prange = range(batch_size)
    if progress:
        prange = tqdm(prange, desc="Plotting")
    for ibatch in prange:
        if ibatch >= cur_example_num:
            break
        # generate figures for each ibatch
        data, arrivals, key, sgram, label = data_batch[ibatch], arrivals_batch[
            ibatch], key_batch[ibatch], sgram_batch[ibatch], label_batch[ibatch]
        # here we assume the data has been procesed
        fig, axes = plt.subplots(7, 1, sharex=True, figsize=(
            20, 30), gridspec_kw={'wspace': 0, 'hspace': 0})
        x = np.arange(data.shape[1])/sampling_rate
        # the max of sgram plot is after 5s of P to 10s of P
        p_arrival = min(arrivals)
        vmax = []
        for i in range(3):
            if global_max:
                i = 0
            if p_arrival+sampling_rate * 5 >= 0 and p_arrival+sampling_rate*15 <= sgram.shape[-1]:
                vmax.append(torch.max(sgram[i][:, p_arrival+sampling_rate *
                                               5:p_arrival+sampling_rate*15]))
            else:
                vmax.append(30)
        max_scale = torch.max(torch.abs(data))
        # R component
        axes[0].plot(x, data[0, :], c="black", lw=1, label="R")
        axes[0].legend()
        axes[1].imshow(sgram[0], aspect='auto', cmap="jet", origin='lower',
                       vmin=0, vmax=vmax[0], extent=x_range+freq_range)
        # T component
        axes[2].plot(x, data[1, :], c="black", lw=1, label="T")
        axes[2].legend()
        axes[3].imshow(sgram[1], aspect='auto', cmap="jet", origin='lower',
                       vmin=0, vmax=vmax[1], extent=x_range+freq_range)
        # Z component
        axes[4].plot(x, data[2, :], c="black", lw=1, label="Z")
        axes[4].legend()
        axes[5].imshow(sgram[2], aspect='auto', cmap="jet", origin='lower',
                       vmin=0, vmax=vmax[2], extent=x_range+freq_range)
        # phases
        color = cm.rainbow(np.linspace(0, 1, len(phases)))
        for i, each_phase in enumerate(phases):
            axes[6].plot(x, label[i+1, :].numpy(), '--',
                         c=color[i], label=each_phase[1:])
            for idx in [0, 2, 4]:
                if 0 < arrivals[i] < sgram.shape[-1]:
                    axes[idx].vlines(x=arrivals[i]/sampling_rate, ymin=-max_scale,
                                     ymax=max_scale, colors=color[i], ls='--', lw=1)
                axes[idx].margins(0)
                axes[idx].set_ylabel('Amplitude', fontsize=18)
            for idx in [1, 3, 5]:
                if 0 < arrivals[i] < sgram.shape[-1]:
                    axes[idx].vlines(x=arrivals[i]/sampling_rate, ymin=freq_range[0],
                                     ymax=freq_range[1], colors=color[i], ls='--', lw=1)
                axes[idx].set_ylabel('Frequency (HZ)', fontsize=18)
        axes[6].plot(x, label[0, :].numpy(), '--',
                     c="black", label="Noise")
        axes[6].set_xlabel('time (s)', fontsize=24)
        axes[6].legend()

        fig.savefig(join(save_dir, f"{key}.pdf"), bbox_inches='tight')
        plt.close(fig)


def show_info_batch(cfg: Config, save_directory: str, data_loader: DataLoader, predict: Optional[torch.Tensor] = None, example_num: int = 8) -> None:
    """Save pdf showing the results for all the batches

    Args:
        conf (Config): the configuration for the APP
        save_directory (str): the saving directory
        data_loader (DataLoader): the data loader of the dataset to plot
        predict (Optional[torch.Tensor]): the optional prediction tensor (if None, plot target instead)
    """
    # https://stackoverflow.com/questions/42544885/error-when-mkdir-in-multi-threads-in-python
    os.makedirs(save_directory, exist_ok=True)  # race condition free
    batch_size = cfg.train.train_batch_size
    for ibatch, each_batch in enumerate(data_loader):
        if ibatch*batch_size >= example_num:
            continue
        cur_example_num = example_num-ibatch * \
            batch_size if (ibatch+1)*batch_size >= example_num else batch_size
        show_info(each_batch, phases=cfg.data.phases,  save_dir=save_directory, sampling_rate=cfg.spectrogram.sampling_rate, x_range=[0, cfg.preprocess.win_length], freq_range=[
                  cfg.spectrogram.freqmin, cfg.spectrogram.freqmax], progress=False, global_max=False, cur_example_num=cur_example_num, predict=predict[ibatch] if predict else None)
