"""
visualize.py

helper functions to visualzie the dataset and model.
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import cm


class VisualizeInfo:
    def __init__(self, phases: List[str], sampling_rate: int, x_range: List[int], freq_range: List[int], global_max: bool = False, sgram_threshold: Optional[int] = None) -> None:
        self.phases = phases
        self.sampling_rate = sampling_rate
        self.x_range = x_range
        self.freq_range = freq_range
        self.global_max = global_max
        self.sgram_threshold = sgram_threshold

    def __call__(self, input_batch: Dict, sgram_batch: torch.Tensor,  predict_batch: torch.Tensor, peaks_batch: Dict[str, List[List[List]]], cur_example_num: int = 0) -> Optional[List[plt.Figure]]:
        if cur_example_num == 0:
            return None
        figs = []
        # * load data
        data_batch: torch.Tensor = input_batch['data'].detach().cpu()
        arrivals_batch: torch.Tensor = input_batch["arrivals"].detach().cpu()
        key_batch: List[str] = input_batch["key"]
        label_batch: torch.Tensor = input_batch["label"].detach().cpu()
        sgram_batch = sgram_batch.detach().cpu()
        predict_batch = predict_batch.detach().cpu()

        # * plot each batch
        batch_size = data_batch.shape[0]
        for ibatch in range(min(batch_size, cur_example_num)):
            # * prepare
            # generate figures for each ibatch
            data, arrivals, key, sgram, label, predict = data_batch[ibatch], arrivals_batch[
                ibatch], key_batch[ibatch], sgram_batch[ibatch], label_batch[ibatch], predict_batch[ibatch]
            peaks_idx, peaks_val = peaks_batch["arrivals"][ibatch], peaks_batch["amps"][ibatch]
            # here we assume the data has been procesed
            fig, axes = plt.subplots(8, 1, sharex=True, figsize=(
                20, 34), gridspec_kw={'wspace': 0, 'hspace': 0})
            axes[0].set_title(key, fontsize=24)
            x = np.arange(data.shape[1])/self.sampling_rate
            # the max of sgram plot is after 5s of P to 10s of P
            # but we should care if specify the max threshold
            # * max threshold
            vmax = []
            if self.sgram_threshold == None:
                p_arrival = min(arrivals)
                for i in range(3):
                    if self.global_max:
                        i = 0
                    if p_arrival+self.sampling_rate * 5 >= 0 and p_arrival+self.sampling_rate*15 <= sgram.shape[-1]:
                        vmax.append(torch.max(sgram[i][:, p_arrival+self.sampling_rate *
                                                       5:p_arrival+self.sampling_rate*15]))
                    else:
                        vmax.append(30)
            else:
                vmax = [self.sgram_threshold]*3
                # print(torch.max(sgram[2]), "++++")
                # p_arrival = min(arrivals)
                # if p_arrival+self.sampling_rate * 5 >= 0 and p_arrival+self.sampling_rate*15 <= sgram.shape[-1]:
                #     print(torch.max(sgram[2][:, p_arrival+self.sampling_rate *
                #                              5:p_arrival+self.sampling_rate*15]), "@@@@")
            max_scale = torch.max(torch.abs(data))
            # * plot wave and sgram
            # R component
            axes[0].plot(x, data[0, :], c="black", lw=1, label="R")
            axes[0].legend()
            axes[1].imshow(sgram[0], aspect='auto', cmap="jet", origin='lower',
                           vmin=0, vmax=vmax[0], extent=self.x_range+self.freq_range)
            # T component
            axes[2].plot(x, data[1, :], c="black", lw=1, label="T")
            axes[2].legend()
            axes[3].imshow(sgram[1], aspect='auto', cmap="jet", origin='lower',
                           vmin=0, vmax=vmax[1], extent=self.x_range+self.freq_range)
            # Z component
            axes[4].plot(x, data[2, :], c="black", lw=1, label="Z")
            axes[4].legend()
            axes[5].imshow(sgram[2], aspect='auto', cmap="jet", origin='lower',
                           vmin=0, vmax=vmax[2], extent=self.x_range+self.freq_range)
            # * plot predictions and targets
            color = cm.rainbow(np.linspace(0, 1, len(self.phases)))
            for i, each_phase in enumerate(self.phases):
                axes[6].plot(x, label[i+1, :].numpy(), '--',
                             c=color[i], label=each_phase[1:])
                axes[7].plot(x, predict[i+1, :].numpy(), '--',
                             c=color[i], label=each_phase[1:])
                for idx in [0, 2, 4]:
                    if 0 < arrivals[i] < sgram.shape[-1]:
                        axes[idx].vlines(x=arrivals[i]/self.sampling_rate, ymin=-max_scale,
                                         ymax=max_scale, colors=color[i], ls='--', lw=1)
                    axes[idx].margins(0)
                    axes[idx].set_ylabel('Amplitude', fontsize=18)
                for idx in [1, 3, 5]:
                    if 0 < arrivals[i] < sgram.shape[-1]:
                        axes[idx].vlines(x=arrivals[i]/self.sampling_rate, ymin=self.freq_range[0],
                                         ymax=self.freq_range[1], colors=color[i], ls='--', lw=1)
                    axes[idx].set_ylabel('Frequency (HZ)', fontsize=18)
                # * plot peaks
                peaksx, peaksy = peaks_idx[i], peaks_val[i]
                for px, py in zip(peaksx, peaksy):
                    axes[7].scatter(px/self.sampling_rate, py,
                                    s=50, color="k", marker="+")
            axes[6].plot(x, label[0, :].numpy(), '--',
                         c="black", label="Noise")
            axes[7].plot(x, predict[0, :].numpy(), '--',
                         c="black", label="Noise")
            axes[7].set_xlabel('time (s)', fontsize=24)
            axes[7].legend()

            figs.append(fig)
        return figs
