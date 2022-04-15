from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from phasenet.conf.load_conf import Config, SpectrogramConfig
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

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        sgram: torch.Tensor = super().__call__(waveform)
        # sgram: torch.Tensor = self.func(data)
        # we should cut the frequency between freqmin to freqmax
        # the time bin length is nt//hop_length+1, we assume the mod is nt%hop_length==0
        df = (self.sampling_rate/2)/(self.n_fft//2)
        freqmin_pos = round(self.freqmin/df)
        freqmax_pos = round(self.freqmax/df)
        sgram = sgram[..., freqmin_pos:freqmax_pos+1, :-1]
        # resize
        sgram = F.resize(sgram, [self.height, self.width])
        sgram = torch.clamp_max(sgram, self.max_clamp)
        return sgram


class PhaseNetModel(pl.LightningModule):
    def __init__(self, model: nn.Module, conf: Config) -> None:
        super().__init__()
        self.spec_conf = conf.spectrogram
        self.model_conf = conf.model
        self.train_conf = conf.train
        # define the model
        self.sgram_trans = GenSgram(self.spec_conf)
        self.model = model(self.model_conf)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # training_step defined the train loop.
        # It is independent of forward
        wave, label = batch["data"], batch["label"]
        sgram = self.sgram_trans(wave)
        output = self.model(sgram)
        predict = output['predict']
        loss = self._criterion(predict, label)
        # refer to https://github.com/PyTorchLightning/pytorch-lightning/issues/10349
        self.log("Loss/train", loss, on_step=False,
                 on_epoch=True, batch_size=len(wave), prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("Loss/validation", loss, on_step=False,
                 on_epoch=True, batch_size=len(batch['data']), prog_bar=True)
        return loss

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("Test Loss", loss, on_step=False,
                 on_epoch=True, batch_size=len(batch['data']))
        return loss

    def _shared_eval_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        wave, label = batch["data"], batch["label"]
        sgram = self.sgram_trans(wave)
        output = self.model(sgram)
        predict = output['predict']
        loss = self._criterion(predict, label)
        return loss

    def _criterion(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = nn.functional.kl_div(
            torch.nn.functional.log_softmax(inputs, dim=1), target, reduction='batchmean',
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.train_conf.learning_rate, weight_decay=self.train_conf.weight_decay, amsgrad=False
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / self.trainer.estimated_stepping_batches) ** 0.9,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"
            }
        }
