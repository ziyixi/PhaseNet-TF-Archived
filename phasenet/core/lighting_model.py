from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from phasenet.conf.load_conf import Config
from phasenet.core.sgram import GenSgram
from phasenet.utils.visualize import VisualizeInfo
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard import SummaryWriter


class PhaseNetModel(pl.LightningModule):
    def __init__(self, model: nn.Module, conf: Config) -> None:
        super().__init__()
        self.spec_conf = conf.spectrogram
        self.model_conf = conf.model
        self.train_conf = conf.train
        self.visualize = conf.visualize

        # define the model
        self.sgram_trans = GenSgram(self.spec_conf)
        self.model = model(self.model_conf)
        # loggers
        self.show_figs = VisualizeInfo(
            phases=conf.data.phases,
            sampling_rate=conf.spectrogram.sampling_rate,
            x_range=[0, conf.data.win_length],
            freq_range=[conf.spectrogram.freqmin, conf.spectrogram.freqmax],
            global_max=False,
            sgram_threshold=conf.visualize.sgram_threshold,
            cur_example_num=conf.visualize.example_num
        )

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # training_step defined the train loop.
        # It is independent of forward
        wave, label = batch["data"], batch["label"]
        sgram = self.sgram_trans(wave)
        output = self.model(sgram)
        predict = output['predict']
        loss = self._criterion(predict, label)
        # * logging
        # refer to https://github.com/PyTorchLightning/pytorch-lightning/issues/10349
        self.log_dict({"Loss/train": loss, "step": self.current_epoch + 1.0},
                      on_step=False, on_epoch=True, batch_size=len(wave), prog_bar=True)
        self._log_figs_train(batch, batch_idx, sgram, predict)
        # * return misfit
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss, sgram, predict = self._shared_eval_step(batch, batch_idx)
        self.log_dict({"Loss/validation": loss, "step": self.current_epoch + 1.0}, on_step=False,
                      on_epoch=True, batch_size=len(batch['data']), prog_bar=True)
        self._log_figs_val(batch, batch_idx, sgram, predict)
        return loss

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss, sgram, predict = self._shared_eval_step(batch, batch_idx)
        self.log("Test Loss", loss, on_step=False,
                 on_epoch=True, batch_size=len(batch['data']))
        self._log_figs_test(batch, batch_idx, sgram, predict)
        return loss

    def _shared_eval_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        wave, label = batch["data"], batch["label"]
        sgram = self.sgram_trans(wave)
        output = self.model(sgram)
        predict = output['predict']
        loss = self._criterion(predict, label)
        return loss, sgram, predict

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
            lambda x: (1 - x / self._num_training_steps) ** 0.9,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"
            }
        }

    @property
    def _num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        # from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(
            self.trainer._data_connector._train_dataloader_source.dataloader())
        batches = min(batches, limit_batches) if isinstance(
            limit_batches, int) else int(limit_batches * batches)

        effective_accum = self.trainer.accumulate_grad_batches * self.trainer.num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @rank_zero_only
    def _log_figs_train(self, batch: Dict, batch_idx: int, sgram: torch.Tensor, predict: torch.Tensor) -> None:
        if batch_idx == 0 and self.visualize.log_train:
            if (self.current_epoch == self.trainer.max_epochs-1) or (self.visualize.log_epoch and (self.current_epoch+1) % self.visualize.log_epoch == 0):
                predict_freq = torch.nn.functional.softmax(predict, dim=1)
                figs = self.show_figs(batch, sgram, predict_freq)
                tensorboard: SummaryWriter = self.logger.experiment
                if self.current_epoch == self.trainer.max_epochs-1:
                    tag = "train/final"
                elif self.visualize.log_epoch and (self.current_epoch+1) % self.visualize.log_epoch == 0:
                    tag = f"train/epoch{self.current_epoch+1}"
                tensorboard.add_figure(
                    tag, figs, global_step=self.current_epoch+1)

    @rank_zero_only
    def _log_figs_val(self, batch: Dict, batch_idx: int, sgram: torch.Tensor, predict: torch.Tensor) -> None:
        if batch_idx == 0 and self.visualize.log_val:
            if (self.current_epoch == self.trainer.max_epochs-1) or (self.visualize.log_epoch and (self.current_epoch+1) % self.visualize.log_epoch == 0):
                predict_freq = torch.nn.functional.softmax(predict, dim=1)
                figs = self.show_figs(batch, sgram, predict_freq)
                tensorboard: SummaryWriter = self.logger.experiment
                if self.current_epoch == self.trainer.max_epochs-1:
                    tag = "validation/final"
                elif self.visualize.log_epoch and (self.current_epoch+1) % self.visualize.log_epoch == 0:
                    tag = f"validation/epoch{self.current_epoch+1}"
                tensorboard.add_figure(
                    tag, figs, global_step=self.current_epoch+1)

    @rank_zero_only
    def _log_figs_test(self, batch: Dict, batch_idx: int, sgram: torch.Tensor, predict: torch.Tensor) -> None:
        if batch_idx == 0 and self.visualize.log_test:
            predict_freq = torch.nn.functional.softmax(predict, dim=1)
            figs = self.show_figs(batch, sgram, predict_freq)
            tensorboard: SummaryWriter = self.logger.experiment
            tensorboard.add_figure(
                "test/final", figs, global_step=self.current_epoch+1)
