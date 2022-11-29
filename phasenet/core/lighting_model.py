from collections import OrderedDict
from os.path import join
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard import SummaryWriter

from phasenet.conf import Config
from phasenet.core.loss import focal_loss
from phasenet.core.sgram import GenSgram
from phasenet.model.unet import UNet
from phasenet.utils.helper import Normalize
from phasenet.utils.metrics import F1, Precision, Recall
from phasenet.utils.peaks import extract_peaks
from phasenet.utils.visualize import VisualizeInfo


class PhaseNetModel(pl.LightningModule):
    def __init__(self, model: nn.Module, conf: Config) -> None:
        super().__init__()
        # * load confs
        self.conf = conf
        self.spec_conf = conf.spectrogram
        self.model_conf = conf.model
        self.train_conf = conf.train
        self.visualize_conf = conf.visualize

        # * define the model
        self.sgram_trans = GenSgram(self.spec_conf)
        # self.model = model(self.model_conf)
        self.model = None
        if model == UNet:
            self.model = UNet(
                features=self.model_conf.init_features,
                in_cha=self.model_conf.in_channels,
                out_cha=self.model_conf.out_channels,
                first_layer_repeating_cnn=self.model_conf.first_layer_repeating_cnn,
                n_freq=self.model_conf.n_freq,
                ksize_down=self.model_conf.encoder_conv_kernel_size,
                ksize_up=self.model_conf.decoder_conv_kernel_size,
                encoder_decoder_depth=self.model_conf.encoder_decoder_depth,
                calculate_skip_for_encoder=self.model_conf.train_with_spectrogram
            )
        else:
            self.model = model()

        # * figure logger
        self.show_figs = VisualizeInfo(
            phases=conf.data.phases,
            sampling_rate=conf.spectrogram.sampling_rate,
            x_range=[0, conf.data.win_length],
            freq_range=[conf.spectrogram.freqmin, conf.spectrogram.freqmax],
            global_max=False,
            sgram_threshold=conf.visualize.sgram_threshold,
            plot_waveform_based_on=conf.visualize.plot_waveform_based_on
        )
        self.figs_train_store = []
        self.figs_val_store = []
        self.figs_test_store = []

        # * metrics
        metrics_dict = OrderedDict()
        for stage in ["metrics_val", "metrics_test"]:
            metrics_dict[stage] = OrderedDict()
            for iphase, phase in enumerate(conf.data.phases):
                metrics_dict[stage][phase] = OrderedDict()

                metrics_dict[stage][phase]["precision"] = Precision(iphase, int(
                    conf.postprocess.metrics_dt_threshold*conf.spectrogram.sampling_rate), conf.data.width)
                metrics_dict[stage][phase]["recall"] = Recall(iphase, int(
                    conf.postprocess.metrics_dt_threshold*conf.spectrogram.sampling_rate), conf.data.width)
                metrics_dict[stage][phase]["f1"] = F1(iphase, int(
                    conf.postprocess.metrics_dt_threshold*conf.spectrogram.sampling_rate), conf.data.width)

                metrics_dict[stage][phase] = nn.ModuleDict(
                    metrics_dict[stage][phase])
            metrics_dict[stage] = nn.ModuleDict(metrics_dict[stage])
        self.metrics = nn.ModuleDict(metrics_dict)

        # * sgram normalizer
        self.sgram_normalizer = Normalize(
            self.spec_conf.mean, self.spec_conf.std)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss, sgram, predict = self._shared_eval_step(
            batch)
        # * logging
        # refer to https://github.com/PyTorchLightning/pytorch-lightning/issues/10349
        log_content = {"loss_train": loss,
                       "step": self.current_epoch + 1.0}
        self.log_dict(log_content,
                      on_step=False, on_epoch=True, batch_size=len(batch["data"]), sync_dist=True, prog_bar=True)

        peaks = extract_peaks(nn.functional.softmax(predict, dim=1), self.conf.data.phases, self.conf.postprocess.sensitive_heights,
                              self.conf.postprocess.sensitive_distances, self.conf.spectrogram.sampling_rate)
        self._log_figs(batch, batch_idx, sgram, predict,
                       peaks, "train")
        # * return misfit
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss, sgram, predict = self._shared_eval_step(batch)
        log_content = {"loss_val": loss,
                       "step": self.current_epoch + 1.0}
        for phase in self.metrics["metrics_val"]:
            for key in self.metrics["metrics_val"][phase]:
                peaks = extract_peaks(nn.functional.softmax(predict, dim=1), self.conf.data.phases, self.conf.postprocess.sensitive_heights,
                                      self.conf.postprocess.sensitive_distances, self.conf.spectrogram.sampling_rate)
                predict_arrivals = peaks["arrivals"]
                self.metrics["metrics_val"][phase][key](
                    predict_arrivals, batch["arrivals"])
                log_content[f"Metrics/val/{phase}/{key}"] = self.metrics["metrics_val"][phase][key]
        self.log_dict(log_content, on_step=False,
                      on_epoch=True, batch_size=len(batch['data']), sync_dist=True, prog_bar=True)
        self._log_figs(batch, batch_idx, sgram,
                       predict, peaks, "val")
        return loss

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss, sgram, predict = self._shared_eval_step(batch)
        # * note we are logging loss but not self.test_loss, and manually compute/reset test metrics to add hyper parameters
        log_content = {"loss_test": loss}
        for phase in self.metrics["metrics_test"]:
            for key in self.metrics["metrics_test"][phase]:
                peaks = extract_peaks(nn.functional.softmax(predict, dim=1), self.conf.data.phases, self.conf.postprocess.sensitive_heights,
                                      self.conf.postprocess.sensitive_distances, self.conf.spectrogram.sampling_rate)
                predict_arrivals = peaks["arrivals"]
                # use .update to avoid automatically call compute
                # also note log Metrics with on_epoch will use Metrics' reduction, which is what we desire (not planning to mean all precision)
                self.metrics["metrics_test"][phase][key].update(
                    predict_arrivals, batch["arrivals"])
                log_content[f"Metrics/test/{phase}/{key}"] = self.metrics["metrics_test"][phase][key]
        self.log_dict(log_content, on_step=False,
                      on_epoch=True, batch_size=len(batch['data']), sync_dist=True)
        self._log_figs(batch, batch_idx, sgram, predict,
                       peaks, "test")
        if self.conf.postprocess.save_test_step_to_disk:
            self.save_test_steps(f"{batch_idx}.pt", {
                "data": batch["data"],
                "arrivals": batch["arrivals"],
                "label": batch["label"],
                "key": batch["key"],
                "sgram": sgram,
                "loss": loss,
                "predict": predict,
                "predict_arrivals": predict_arrivals
            })

        return loss

    def test_epoch_end(self, outputs: List[torch.Tensor]):
        metrics = {}
        for phase in self.metrics["metrics_test"]:
            for key in self.metrics["metrics_test"][phase]:
                metrics[f"Metrics/{phase}/{key}"] = self.metrics["metrics_test"][phase][key].compute()
                self.metrics["metrics_test"][phase][key].reset()
        # actually not needed as only one epoch is presented
        self.log_hparms(metrics)

    def _shared_eval_step(self, batch: Dict) -> torch.Tensor:
        wave, label = batch["data"], batch["label"]
        sgram_raw = self.sgram_trans(wave)
        sgram = self.sgram_normalizer(sgram_raw)
        if self.model_conf.train_with_spectrogram:
            output = self.model(sgram)
        else:
            wave_with_freq_dimension = wave[..., None, :]
            output = self.model(wave_with_freq_dimension)
        predict = output['predict']
        if self.train_conf.loss_func == "kl_div":
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(predict, dim=1), label, reduction='batchmean',
            )
        elif self.train_conf.loss_func == "focal":
            loss = focal_loss(
                nn.functional.softmax(predict, dim=1), label,
            )
        # sgram_raw should only be for logging
        if self.spec_conf.power == None:
            # convert to power
            real = sgram_raw[:, :3, :, :]
            imag = sgram_raw[:, 3:, :, :]
            sgram_raw = real**2+imag**2
        return loss, sgram_raw, predict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.train_conf.learning_rate, weight_decay=self.train_conf.weight_decay, amsgrad=False
        )
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=self.train_conf.learning_rate, momentum=0.9)
        # lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=1, end_factor=0.1, total_iters=self._num_training_steps)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.train_conf.step_lr_milestones, gamma=self.train_conf.step_lr_gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch"
            }
        }

    # * ============== helpers ============== * #
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

    def log_hparms(self, metrics: Dict[str, torch.Tensor]):
        hparam = {
            "data/stack_ratio": self.conf.data.stack_ratio,
            "data/noise_replace_ratio": self.conf.data.noise_replace_ratio,
            "spectrogram/n_fft": self.conf.spectrogram.n_fft,
            "spectrogram/max_clamp": self.conf.spectrogram.max_clamp,
            "model/init_features": self.conf.model.init_features,
            "model/first_layer_repeating_cnn": self.conf.model.first_layer_repeating_cnn,
            "model/encoder_conv_kernel_size": torch.tensor(self.conf.model.encoder_conv_kernel_size),
            "model/decoder_conv_kernel_size": torch.tensor(self.conf.model.decoder_conv_kernel_size),
            "model/encoder_decoder_depth": self.conf.model.encoder_decoder_depth,
            "train/learning_rate": self.conf.train.learning_rate,
            "train/weight_decay": self.conf.train.weight_decay,
        }
        self.logger.log_hyperparams(hparam, metrics)

    def save_test_steps(self, file_name: str, to_save: Dict[str, torch.Tensor]) -> None:
        # save tensors to disk for further analysis
        file_path = join(self.conf.postprocess.test_step_save_path, file_name)
        # update to_save to cpu array
        to_save_cpu = {}
        for key in to_save:
            if isinstance(to_save[key], torch.Tensor):
                to_save_cpu[key] = to_save[key].detach().to(
                    dtype=torch.float32).cpu()
            else:
                to_save_cpu[key] = to_save[key]
        torch.save(to_save_cpu, file_path)

    # * ============== figure plotting ============== * #
    @rank_zero_only
    def _log_figs(self, batch: Dict, batch_idx: int, sgram: torch.Tensor, predict: torch.Tensor, peaks: Dict[str, List[List[List]]], stage: str) -> None:
        if_log = {
            "train": self.visualize_conf.log_train,
            "val": self.visualize_conf.log_val,
            "test": self.visualize_conf.log_test
        }
        figs_store = {
            "train": self.figs_train_store,
            "val": self.figs_val_store,
            "test": self.figs_test_store
        }
        if not if_log[stage]:
            return
        if ((self.current_epoch == self.trainer.max_epochs-1) or (self.visualize_conf.log_epoch and (self.current_epoch+1) % self.visualize_conf.log_epoch == 0)) or stage == "test":
            batch_size = len(sgram)
            finished_examples = batch_size*batch_idx
            if finished_examples < self.visualize_conf.example_num:
                if finished_examples+batch_size < self.visualize_conf.example_num:
                    example_this_batch = batch_size
                    last_step = False
                else:
                    example_this_batch = self.visualize_conf.example_num-finished_examples
                    last_step = True

                figs = self.show_figs(
                    batch, sgram, nn.functional.softmax(predict, dim=1), peaks, example_this_batch)
                figs_store[stage].extend(figs)
                if last_step:
                    tensorboard: SummaryWriter = self.logger.experiment
                    # tag name
                    if self.current_epoch == self.trainer.max_epochs-1 or stage == "test":
                        tag = f"{stage}/final"
                    elif self.visualize_conf.log_epoch and (self.current_epoch+1) % self.visualize_conf.log_epoch == 0:
                        tag = f"{stage}/epoch{self.current_epoch+1}"
                    # save figures in dir when asked in the test stage
                    if self.visualize_conf.log_test_seprate_folder and stage == "test":
                        for idx, each_fig in enumerate(self.figs_test_store):
                            each_fig.savefig(
                                join(self.visualize_conf.log_test_seprate_folder_path, f"{idx+1}.pdf"))
                    tensorboard.add_figure(
                        tag, figs_store[stage], global_step=self.current_epoch+1)
                    figs_store[stage].clear()


class MeanStdEstimator(pl.LightningModule):
    def __init__(self, conf: Config) -> None:
        super().__init__()
        self.spec_conf = conf.spectrogram
        self.sgram_trans = GenSgram(self.spec_conf)
        self.dummy_layer = nn.BatchNorm2d(conf.model.in_channels)

    def training_step(self, batch: Dict, batch_idx: int):
        wave = batch["data"]
        sgram = self.sgram_trans(wave)

        mean = sgram.mean(dim=[0, 2, 3])
        meansq = (sgram**2).mean(dim=[0, 2, 3])
        return {
            "loss": self.dummy_layer(sgram).mean(),
            "mean": mean,
            "meansq": meansq
        }

    def training_epoch_end(self, outputs: Dict[str, torch.tensor]) -> None:
        mean = torch.stack([x["mean"] for x in outputs]).mean(dim=0)
        meansq = torch.stack([x["meansq"] for x in outputs]).mean(dim=0)
        std = torch.sqrt(meansq - mean**2)
        if self.global_rank == 0:
            print("="*10)
            print(f"mean: {mean}")
            print(f"std: {std}")
            print("="*10)

    def configure_optimizers(self):
        # dummy one, not actually important
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=0.01, weight_decay=0.001, amsgrad=False
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 90, 120], gamma=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch"
            }
        }
