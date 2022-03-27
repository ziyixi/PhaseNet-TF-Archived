"""
visualize.py

helper functions to visualzie the dataset and model.
"""
from os.path import join
from typing import Dict, List, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import cm
from phasenet.utils.normalize import normalize_range
from tqdm import tqdm


def show_data_batch(data_batch: Dict, phases: List[str], save_dir: str, sampling_rate: int, show_output: bool = False, output_batch: Optional[torch.Tensor] = None) -> None:
    """
    plot the input dataset batch after applying at least ArrivalToArray
    """
    data_all: torch.Tensor = data_batch['data']
    label_all: torch.Tensor = data_batch['label']
    key_all: List[str] = data_batch['key']
    arrivals_all: torch.Tensor = data_batch['arrivals']
    batch_size = data_all.shape[0]
    # plot three components with target
    for index in range(batch_size):
        # plot single
        data, label, key, arrival = data_all[index], label_all[index], key_all[index], arrivals_all[index]
        rows = 4
        if show_output:
            rows += 1
        fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(20, 20))
        x = np.arange(data.shape[1])/sampling_rate
        axes[0].plot(x, normalize_range(
            data[0, :].numpy(), -10, 10), c="black", lw=1, label="R")
        axes[0].set_ylabel("Amplitude")
        axes[1].plot(x, normalize_range(
            data[1, :].numpy(), -10, 10), c="black", lw=1, label="T")
        axes[1].set_ylabel("Amplitude")
        axes[2].plot(x, normalize_range(
            data[2, :].numpy(), -10, 10), c="black", lw=1, label="Z")
        axes[2].set_ylabel("Amplitude")
        # phases
        color = cm.rainbow(np.linspace(0, 1, len(phases)))
        for i, each_phase in enumerate(phases):
            axes[3].plot(x, label[i+1, :].numpy(), '--',
                         c=color[i], label=each_phase[1:])
            for idx in range(3):
                axes[idx].vlines(x=arrival[i]/sampling_rate, ymin=-10, ymax=10,
                                 colors=color[i], ls='--', lw=2)
        axes[3].set_ylabel("Target Propability")
        axes[3].set_xlabel("Time (s)")
        # output
        if show_output:
            output = output_batch[index]
            for i, each_phase in enumerate(phases):
                axes[4].plot(x, output[i+1, :].detach().numpy(), '--',
                             c=color[i], label=each_phase[1:])
            axes[4].set_ylabel("Estimated Propability")
            axes[4].set_xlabel("Time (s)")

        for i in range(rows):
            axes[i].legend()

        fig.savefig(join(save_dir, f"{key}.pdf"), bbox_inches='tight')


def show_sgram(sgram: torch.Tensor, key_all: List[str], save_dir: str):
    batch_size, nc, nt, nf = sgram.shape
    for index in range(batch_size):
        fig, axes = plt.subplots(
            nc//2, 2, sharex=True, sharey=True, figsize=(20, 8))
        for ic in range(nc):
            axes[ic//2][ic % 2].imshow(sgram[index, ic].transpose(-1, -
                                                                  2).numpy(), interpolation='bilinear')
        fig.savefig(
            join(save_dir, f"{key_all[index]}.sgram.pdf"), bbox_inches='tight')


class BatchInput(TypedDict):
    data: torch.Tensor
    arrivals: torch.Tensor
    key: List[str]
    sgram: torch.Tensor
    label: torch.Tensor


def show_input(input_batch: BatchInput, phases: List[str], save_dir: str, sampling_rate: int, x_range: List[int], freq_range: List[int], merge: bool = False, global_max: bool = False, progress: bool = False) -> None:
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
        progress (bool): if show the progress bar
    """
    data_batch, arrivals_batch, key_batch, sgram_batch, label_batch = input_batch[
        'data'], input_batch["arrivals"], input_batch["key"], input_batch["sgram"], input_batch["label"]
    batch_size = data_batch.shape[0]
    prange = range(batch_size)
    if progress:
        prange = tqdm(prange, desc="Plotting")
    for ibatch in prange:
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
            vmax.append(torch.max(sgram[i][:, p_arrival+sampling_rate *
                                  5:p_arrival+sampling_rate*15]))
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
                axes[idx].vlines(x=arrivals[i]/sampling_rate, ymin=-max_scale,
                                 ymax=max_scale, colors=color[i], ls='--', lw=1)
                axes[idx].margins(0)
                axes[idx].set_ylabel('Amplitude', fontsize=18)
            for idx in [1, 3, 5]:
                axes[idx].vlines(x=arrivals[i]/sampling_rate, ymin=freq_range[0],
                                 ymax=freq_range[1], colors=color[i], ls='--', lw=1)
                axes[idx].set_ylabel('Frequency (HZ)', fontsize=18)
        axes[6].set_xlabel('time (s)', fontsize=24)
        axes[6].legend()

        fig.savefig(join(save_dir, f"{key}.pdf"), bbox_inches='tight')
        plt.close(fig)
