"""
transforms.py

custome transforms applied to waveform dataset.
"""
from typing import Dict

import torch
import torchvision.transforms.functional as F
from torchaudio.transforms import Spectrogram
from phasenet.conf.load_conf import PreprocessConfig, SpectrogramConfig


class RandomShift:
    def __init__(self, cfg_preprocess: PreprocessConfig) -> None:
        self.width = cfg_preprocess.width

    def __call__(self, sample: Dict) -> Dict:
        sample_data = self.shift(sample, "data", "arrivals")
        sample_random_data = self.shift(
            sample_data, "random_data", "random_arrivals")
        return sample_random_data

    def shift(self, sample: Dict, data_key: str, arrivals_key: str) -> Dict:
        sample_updated = sample.copy()
        data, arrivals = sample_updated[data_key], sample_updated[arrivals_key]
        # determine the shift range
        left_bound = -torch.max(arrivals)
        right_bound = self.width-torch.min(arrivals)
        # update arrivals
        shift = torch.randint(left_bound, right_bound, (1,)).item()
        arrivals_shifted = arrivals.clone()
        for i in range(len(arrivals)):
            arrivals_shifted[i] += shift
        # update data
        data_shifted = data.roll(shift, dims=1)
        if shift >= 0:
            data_shifted[:, :shift] = 0
        else:
            data_shifted[:, shift:] = 0
        sample_updated.update({
            data_key: data_shifted,
            arrivals_key: arrivals_shifted
        })
        return sample_updated


class GenLabel:
    def __init__(self, cfg_preprocess: PreprocessConfig) -> None:
        self.label_shape = cfg_preprocess.label_shape
        self.label_width = cfg_preprocess.label_width

    def __call__(self, sample: Dict) -> Dict:
        sample_data = self.label(sample, "data", "arrivals", "label")
        sample_random_data = self.label(
            sample_data, "random_data", "random_arrivals", "random_label")
        return sample_random_data

    def label(self, sample: Dict, data_key: str, arrivals_key: str, label_key: str) -> Dict:
        sample_updated = sample.copy()
        data, arrivals = sample_updated[data_key], sample_updated[arrivals_key]
        res = torch.zeros(len(arrivals)+1, data.shape[1])
        if self.label_shape == "gaussian":
            label_window = torch.exp(-(torch.arange(-self.label_width //
                                     2, self.label_width//2+1))**2/(2*(self.label_width/6)**2))
        elif self.label_shape == "triangle":
            label_window = 1 - \
                torch.abs(
                    2/self.label_width * (torch.arange(-self.label_width//2, self.label_width//2+1)))
        else:
            raise Exception(
                f"label shape {self.label_shape} is not supported!")

        # the first class set as noise
        for i, idx in enumerate(arrivals):
            # the index for arrival times
            start = idx-self.label_width//2
            end = idx+self.label_width//2+1
            if start >= 0 and end <= res.shape[1]:
                res[i+1, start:end] = label_window
        # can sum as the first row is 0
        res[0, :] = 1-torch.sum(res, 0)
        sample_updated.update({
            label_key: res
        })
        return sample_updated


class StackRand:
    def __init__(self, cfg_preprocess: PreprocessConfig) -> None:
        self.stack_ratio = cfg_preprocess.stack_ratio
        self.min_stack_gap = cfg_preprocess.min_stack_gap

    def __call__(self, sample: Dict) -> Dict:
        # * stack data / label
        sample_updated = sample.copy()
        arrivals, random_arrivals, data, random_data, label, random_label = sample_updated['arrivals'], sample_updated['random_arrivals'], sample_updated[
            'data'], sample_updated['random_data'], sample_updated['label'], sample_updated['random_label']
        # if arrivals overlap, skip
        for i in range(len(arrivals)):
            if torch.abs(arrivals[i]-random_arrivals[i]) <= self.min_stack_gap:
                return sample_updated
        # if random larger than ratio, skip
        if torch.rand(1).item() > self.stack_ratio:
            return sample_updated
        # handle stacking
        stack_data = data+random_data
        stack_label = label+random_label
        # * here we have to scale the poss to 1 for signals and recalculate the noise
        stack_label = torch.clamp_max(stack_label, 1.0)
        stack_label[0, :] = 1-torch.sum(stack_label[1:], 0)

        sample_updated.update({
            'data': stack_data,
            'label': stack_label
        })
        return sample_updated


class GenSgram(Spectrogram):
    def __init__(self, cfg_spectrogram: SpectrogramConfig, device: torch.device) -> None:
        # since Spectrogram has no params, we don't need to set it as no_grad
        window_fn_maper = {
            "hann": torch.hann_window
        }
        if cfg_spectrogram.window_fn in window_fn_maper:
            win = window_fn_maper[cfg_spectrogram.window_fn]
        else:
            raise Exception(
                f"Unsupportd windows func {cfg_spectrogram.window_fn}. Avaliable candiates are {list(window_fn_maper.keys())}. Try to use the supported func name instead.")
        super().__init__(n_fft=cfg_spectrogram.n_fft,
                         hop_length=cfg_spectrogram.hop_length, power=cfg_spectrogram.power, window_fn=win)
        super().to(device)
        # self.func = partial(spectrogram, n_fft=n_fft,
        #                     hop_length=hop_length, power=power, window_fn=win)
        self.n_fft = cfg_spectrogram.n_fft
        self.freqmin = cfg_spectrogram.freqmin
        self.freqmax = cfg_spectrogram.freqmax
        self.sampling_rate = cfg_spectrogram.sampling_rate
        self.height = cfg_spectrogram.height
        self.width = cfg_spectrogram.width
        self.max_clamp = cfg_spectrogram.max_clamp
        self.device = device

    def __call__(self, sample: Dict) -> Dict:
        sample_updated = sample.copy()
        data: torch.Tensor = sample_updated['data'].to(self.device)
        sgram: torch.Tensor = super().__call__(data)
        # sgram: torch.Tensor = self.func(data)
        # we should cut the frequency between freqmin to freqmax
        # the time bin length is nt//hop_length+1, we assume the mod is nt%hop_length==0
        df = (self.sampling_rate/2)/(self.n_fft//2)
        freqmin_pos = round(self.freqmin/df)
        freqmax_pos = round(self.freqmax/df)
        sgram = sgram[:, freqmin_pos:freqmax_pos+1, :-1]
        # resize
        sgram = F.resize(sgram, [self.height, self.width])
        sgram = torch.clamp_max(sgram, self.max_clamp)
        sample_updated.update({
            'sgram': sgram
        })
        return sample_updated


class ScaleAmp:
    def __init__(self, cfg_preprocess: PreprocessConfig) -> None:
        self.max_amp = cfg_preprocess.scale_max_amp
        self.global_max = cfg_preprocess.scale_global_max

    def __call__(self, sample: Dict) -> Dict:
        sample_data = self.scale(sample, "data")
        sample_random_data = self.scale(sample_data, "random_data")
        return sample_random_data

    def scale(self, sample: Dict, key: str) -> Dict:
        sample_updated = sample.copy()
        data: torch.Tensor = sample_updated[key].clone()
        if self.global_max:
            raw_max = torch.max(torch.abs(data))
            data = data/raw_max
        else:
            for ich in range(data.shape[0]):
                raw_max = torch.max(torch.abs(data[ich]))
                data[ich, :] = data[ich, :]/raw_max
        sample_updated.update({
            key: data
        })
        return sample_updated
