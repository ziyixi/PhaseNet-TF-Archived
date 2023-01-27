from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from phasenet.conf import DataConfig

from .dataset import WaveFormDataset
from .transforms import (GenLabel, RandomShift, ReplaceNoise, ScaleAmp,
                         StackRand)


class WaveFormDataModule(pl.LightningDataModule):
    def __init__(self, data_conf: DataConfig, run_type: str):
        super().__init__()
        self.data_conf = data_conf
        self.run_type = run_type
        self.trans = {
            "scale": ScaleAmp(data_conf),
            "shift": RandomShift(data_conf),
            "label": GenLabel(data_conf),
            "stack": StackRand(data_conf),
            "replace_noise": ReplaceNoise(data_conf),
        }

    def prepare_data(self):
        # cache
        WaveFormDataset(self.data_conf, data_type="train", prepare=True)
        WaveFormDataset(self.data_conf, data_type="val", prepare=True)
        if self.run_type != "hyper_tune":
            WaveFormDataset(self.data_conf, data_type="test", prepare=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            transform_train = Compose([self.trans[key]
                                       for key in self.data_conf.train_trans if key not in ["stack", "replace_noise"]])
            transform_val = Compose([self.trans[key]
                                     for key in self.data_conf.val_trans if key not in ["stack", "replace_noise"]])
            stack_transform_train = self.trans["stack"] if self.data_conf.stack else None
            replace_noise_transform_train = self.trans["replace_noise"] if self.data_conf.replace_noise else None
            scale_at_end_transform_train = self.trans["scale"] if self.data_conf.scale_at_end else None

            self.wave_train = WaveFormDataset(
                self.data_conf, data_type="train", transform=transform_train, stack_transform=stack_transform_train, replace_noise_transform=replace_noise_transform_train, scale_at_end_transform=scale_at_end_transform_train)
            self.wave_val = WaveFormDataset(
                self.data_conf, data_type="val", transform=transform_val)

        if stage == "test" or stage is None:
            if self.run_type != "hyper_tune":
                transform = Compose([self.trans[key]
                                    for key in self.data_conf.test_trans if key not in ["stack", "replace_noise"]])
                self.wave_test = WaveFormDataset(
                    self.data_conf, data_type="test", transform=transform)
            else:
                transform = Compose([self.trans[key]
                                    for key in self.data_conf.val_trans if key not in ["stack", "replace_noise"]])
                self.wave_test = WaveFormDataset(
                    self.data_conf, data_type="val", transform=transform)

    def train_dataloader(self):
        return DataLoader(self.wave_train, batch_size=self.data_conf.train_batch_size, shuffle=self.data_conf.train_shuffle, num_workers=self.data_conf.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.wave_val, batch_size=self.data_conf.val_batch_size, shuffle=False, num_workers=self.data_conf.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.wave_test, batch_size=self.data_conf.test_batch_size, shuffle=False, num_workers=self.data_conf.num_workers, pin_memory=True, persistent_workers=True)
