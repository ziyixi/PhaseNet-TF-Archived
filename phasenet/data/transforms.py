"""
transforms.py

custome transforms applied to waveform dataset.
"""
from typing import Dict

import torch
import torchvision.transforms.functional as F
from torchaudio.transforms import Spectrogram


class RandomShift:
    def __init__(self, width: int, buffer_width: int) -> None:
        self.width = width
        self.buffer_width = buffer_width

    def __call__(self, sample: Dict) -> Dict:
        data, arrivals = sample['data'], sample['arrivals']
        # determine the shift range
        left_bound = torch.min(arrivals)
        if left_bound < self.buffer_width:
            left_bound = 0
        else:
            left_bound = -(left_bound-self.buffer_width)
        right_bound = self.width-torch.max(arrivals)
        if right_bound < self.buffer_width:
            right_bound = 0
        else:
            right_bound = right_bound-self.buffer_width
        # update arrivals
        shift = torch.randint(left_bound, right_bound, (1,)).item()
        for i in range(len(arrivals)):
            arrivals[i] += shift
        # update data
        data = data.roll(shift, dims=1)
        sample.update({
            'data': data,
            'arrivals': arrivals
        })
        return sample


class GenLabel:
    def __init__(self, label_shape: str = "gaussian", label_width: int = 120) -> None:
        self.label_shape = label_shape
        self.label_width = label_width

    def __call__(self, sample: Dict) -> Dict:
        data, arrivals = sample['data'], sample['arrivals']
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
        res[0, :] = 1-torch.sum(res, 0)
        sample.update({
            "data": data,
            "label": res,
            "arrivals": arrivals
        })
        return sample


class GenSgram(Spectrogram):
    def __init__(self, n_fft: int, hop_length: int, power: int, window_fn: str, freqmin: float, freqmax: float, sampling_rate: int, height: int, width: int, max_clamp: int, device: torch.device) -> None:
        # since Spectrogram has no params, we don't need to set it as no_grad
        window_fn_maper = {
            "hann": torch.hann_window
        }
        if window_fn in window_fn_maper:
            win = window_fn_maper[window_fn]
        else:
            raise Exception(
                f"Unsupportd windows func {window_fn}. Avaliable candiates are {list(window_fn_maper.keys())}. Try to use the supported func name instead.")
        super().__init__(n_fft=n_fft, hop_length=hop_length, power=power, window_fn=win)
        super().to(device)
        # self.func = partial(spectrogram, n_fft=n_fft,
        #                     hop_length=hop_length, power=power, window_fn=win)
        self.n_fft = n_fft
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.sampling_rate = sampling_rate
        self.height = height
        self.width = width
        self.max_clamp = max_clamp
        self.device = device

    def __call__(self, sample: Dict) -> Dict:
        data: torch.Tensor = sample['data'].to(self.device)
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
        sample.update({
            'sgram': sgram
        })
        return sample


class ScaleAmp:
    def __init__(self, max_amp: float = 1, global_max: bool = True) -> None:
        self.max_amp = max_amp
        self.global_max = global_max

    def __call__(self, sample: Dict) -> Dict:
        data: torch.Tensor = sample['data']
        if self.global_max:
            raw_max = torch.max(torch.abs(data))
            data = data/raw_max
        else:
            for ich in range(data.shape[0]):
                raw_max = torch.max(torch.abs(data[ich]))
                data[ich, :] = data[ich, :]/raw_max
        sample['data'] = data
        return sample
