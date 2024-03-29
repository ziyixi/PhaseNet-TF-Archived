"""
visualize.py

helper functions to visualzie the dataset and model.
"""
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import cm
from obspy.core.trace import Trace


class VisualizeInfo:
    def __init__(self, phases: List[str], sampling_rate: int, x_range: List[int], freq_range: List[int], global_max: bool = False, sgram_threshold: Optional[int] = None, plot_waveform_based_on: str = "all") -> None:
        self.phases = phases
        self.sampling_rate = sampling_rate
        self.x_range = x_range
        self.freq_range = freq_range
        self.global_max = global_max
        self.sgram_threshold = sgram_threshold
        self.plot_waveform_based_on = plot_waveform_based_on
        self.ps_idx = None
        for iphase, phase in enumerate(self.phases):
            if phase == "TPS":
                self.ps_idx = iphase
                break

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
            # * plot sgram
            # R component
            axes[1].imshow(sgram[0], aspect='auto', cmap="jet", origin='lower',
                           vmin=0, vmax=vmax[0], extent=self.x_range+self.freq_range)
            # T component
            axes[3].imshow(sgram[1], aspect='auto', cmap="jet", origin='lower',
                           vmin=0, vmax=vmax[1], extent=self.x_range+self.freq_range)
            # Z component
            axes[5].imshow(sgram[2], aspect='auto', cmap="jet", origin='lower',
                           vmin=0, vmax=vmax[2], extent=self.x_range+self.freq_range)
            # * ps freq line (raw)
            if "ps_freqs" in input_batch:
                for iax in [1, 3, 5]:
                    axes[iax].hlines(y=input_batch["ps_freqs"][ibatch].detach().cpu(), xmin=0,
                                     xmax=sgram.shape[-1]/self.sampling_rate, colors="w", ls='-', lw=1)
            # * ps freq range line (predict)
            # * should plot max ps loc
            # * ps x range -2s +5s
            fs, fe = None, None
            if self.ps_idx and len(peaks_idx[self.ps_idx]) > 0:
                freq_win_length = 12
                freq_range = [10, 64]
                ps_idx = peaks_idx[self.ps_idx][np.argmax(
                    peaks_val[self.ps_idx])]
                ps_idx_start = ps_idx-int(1*self.sampling_rate)
                ps_idx_end = ps_idx+int(3*self.sampling_rate)
                noise_idx_start = ps_idx-int(5*self.sampling_rate)
                noise_idx_end = ps_idx-int(1*self.sampling_rate)
                if ps_idx_start < 0:
                    ps_idx_start = 0
                if ps_idx_end > sgram.shape[-1]:
                    ps_idx_end = sgram.shape[-1]
                fs, fe = spectrogram_extract_ps_freq(
                    sgram, ps_idx_start, ps_idx_end, freq_range, freq_win_length, noise_idx_start, noise_idx_end)
                fs = fs / \
                    sgram.shape[-2]*(self.freq_range[1] -
                                     self.freq_range[0])+self.freq_range[0]
                fe = fe / \
                    sgram.shape[-2]*(self.freq_range[1] -
                                     self.freq_range[0])+self.freq_range[0]
                # plot
                for iax in [1, 3, 5]:
                    axes[iax].hlines(y=fs, xmin=0,
                                     xmax=sgram.shape[-1]/self.sampling_rate, colors="w", ls='--', lw=1)
                    axes[iax].hlines(y=fe, xmin=0,
                                     xmax=sgram.shape[-1]/self.sampling_rate, colors="w", ls='--', lw=1)

            # * plot wave
            # put it here as we may need PS filter range
            fig_label = {
                "all": " (no further filtering)",
                "P": " (0.2->5 HZ)",
                "PS": " (dynamic based on PS)"
            }

            filtered, status = self.filter_waveform(
                data[0, :], fs, fe)
            axes[0].plot(x, filtered, c="black", lw=1,
                         label="R"+fig_label[status])
            axes[0].legend()
            filtered, status = self.filter_waveform(
                data[1, :], fs, fe)
            axes[2].plot(x, filtered, c="black", lw=1,
                         label="T"+fig_label[status])
            axes[2].legend()
            filtered, status = self.filter_waveform(
                data[2, :], fs, fe)
            axes[4].plot(x, filtered, c="black", lw=1,
                         label="Z"+fig_label[status])
            axes[4].legend()

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

    def filter_waveform(self, data: torch.Tensor, fs: Optional[float] = None, fe: Optional[float] = None) -> Tuple[torch.Tensor, str]:
        if self.plot_waveform_based_on == "all":
            return data, "all"

        wave = Trace(data=data.numpy())
        wave.stats.sampling_rate = self.sampling_rate
        # filter based on the freq range info
        # P 0.2 -> 5 HZ, PS: dynamic
        if self.plot_waveform_based_on == "P":
            wave.filter('bandpass', freqmin=0.2, freqmax=5,
                        corners=4, zerophase=False)
        elif self.plot_waveform_based_on == "PS":
            if fs == None or fe == None:
                # no fs/fe. no PS picks, return origional waveform
                return data, "all"
            else:
                wave.filter('bandpass', freqmin=fs, freqmax=fe,
                            corners=4, zerophase=False)
        else:
            raise Exception(f"no support for {self.plot_waveform_based_on=}")
        return torch.tensor(wave.data), self.plot_waveform_based_on


def spectrogram_extract_ps_freq(sgram_all_phases: torch.Tensor, y_start: int, y_end: int, x_range: List[int], x_length: int, noise_idx_start: int, noise_idx_end: int):
    # * given sgram, and y (time) indexes start and end, find x (freq) indexes start and end
    # sgram = sgram_all_phases.sum(axis=0)
    # we want to only consider the R component when finding PS
    # 0:R, 1:T, 2:Z  Since we are looking at PS, we should use the horizontal component, so it should be some l2 mean
    sgram = torch.sqrt(sgram_all_phases[0]**2+sgram_all_phases[1]**2)
    themax = 0
    s, e = 0, 0
    for x_start in range(x_range[0], x_range[1]):
        cur = sgram[x_start:x_start+x_length, y_start:y_end].sum()
        if cur > themax:
            # s, e = x_start, x_start+x_length
            themax = cur
    # now the range is s->e, value is the max
    # we only consider the range max/10 -> max, but with best SNR
    ratio_global = float("-inf")
    for x_start in range(x_range[0], x_range[1]):
        cur = sgram[x_start:x_start+x_length, y_start:y_end].sum()
        noise = sgram[x_start:x_start+x_length,
                      noise_idx_start:noise_idx_end].sum()
        if cur >= themax/2:
            cur_level = cur/(y_end-y_start)
            noise_level = noise/(noise_idx_end-noise_idx_start)
            ratio = cur_level/noise_level
            if ratio > ratio_global:
                ratio_global = ratio
                s, e = x_start, x_start+x_length
    return s, e
