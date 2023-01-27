import torch
import torchvision.transforms.functional as F
from phasenet.conf import SpectrogramConfig
from torchaudio.transforms import Spectrogram


class GenSgram(Spectrogram):
    def __init__(self, spec_conf: SpectrogramConfig) -> None:
        # since Spectrogram has no params, we don't need to set it as no_grad
        window_fn_maper = {
            "hann": torch.hann_window
        }
        if spec_conf.window_fn in window_fn_maper:
            win = window_fn_maper[spec_conf.window_fn]
        else:
            raise Exception(
                f"Unsupportd windows func {spec_conf.window_fn}. Avaliable candiates are {list(window_fn_maper.keys())}. Try to use the supported func name instead.")
        super().__init__(n_fft=spec_conf.n_fft, hop_length=spec_conf.hop_length,
                         power=spec_conf.power, window_fn=win)
        self.n_fft = spec_conf.n_fft
        self.freqmin = spec_conf.freqmin
        self.freqmax = spec_conf.freqmax
        self.sampling_rate = spec_conf.sampling_rate
        self.height = spec_conf.height
        self.width = spec_conf.width
        self.max_clamp = spec_conf.max_clamp
        self.power = spec_conf.power

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        sgram: torch.Tensor = super().__call__(waveform)
        # sgram: torch.Tensor = self.func(data)
        # we should cut the frequency between freqmin to freqmax
        # the time bin length is nt//hop_length+1, we assume the mod is nt%hop_length==0
        df = (self.sampling_rate/2)/(self.n_fft//2)
        freqmin_pos = round(self.freqmin/df)
        freqmax_pos = round(self.freqmax/df)
        sgram = sgram[..., freqmin_pos:freqmax_pos+1, :-1]
        if self.power == 2:
            sgram = F.resize(sgram, [self.height, self.width])
            sgram = torch.clamp_max(sgram, self.max_clamp)
        elif self.power == None:
            # first 3 channel as real, last 3 as imag
            real = sgram.real
            imag = sgram.imag
            p = sgram.abs()**2+0.001
            ratio = torch.clamp_max(p, self.max_clamp)/p
            # ! note, this can only be done with batch dimension
            sgram = torch.cat([real*ratio, imag*ratio], dim=1)
            # resize seems not keeping imag
            sgram = F.resize(sgram, [self.height, self.width])
        else:
            raise Exception(f"spec power {self.power} is not implemented yet!")
        return sgram
