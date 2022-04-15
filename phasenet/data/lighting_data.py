from typing import Optional

import pytorch_lightning as pl
from phasenet.conf.load_conf import DataConfig
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

from .dataset import WaveFormDataset
from .transforms import GenLabel, RandomShift, ScaleAmp, StackRand


class WaveFormDataModule(pl.LightningDataModule):
    def __init__(self, data_conf: DataConfig):
        super().__init__()
        self.data_conf = data_conf
        self.trans = {
            "scale": ScaleAmp(data_conf),
            "shift": RandomShift(data_conf),
            "label": GenLabel(data_conf),
            "stack": StackRand(data_conf),
        }

    def prepare_data(self):
        # cache
        WaveFormDataset(self.data_conf, data_type="train", prepare=True)
        WaveFormDataset(self.data_conf, data_type="val", prepare=True)
        WaveFormDataset(self.data_conf, data_type="test", prepare=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            transform_train = Compose([self.trans[key]
                                       for key in self.data_conf.train_trans if key != "stack"])
            transform_val = Compose([self.trans[key]
                                     for key in self.data_conf.val_trans if key != "stack"])
            stack_transform_train = self.trans["stack"] if (
                "stack" in self.data_conf.train_trans) else None
            stack_transform_val = self.trans["stack"] if (
                "stack" in self.data_conf.val_trans) else None

            self.wave_train = WaveFormDataset(
                self.data_conf, data_type="train", transform=transform_train, stack_transform=stack_transform_train)
            self.wave_val = WaveFormDataset(
                self.data_conf, data_type="val", transform=transform_val, stack_transform=stack_transform_val)

        if stage == "test" or stage is None:
            transform = Compose([self.trans[key]
                                for key in self.data_conf.test_trans if key != "stack"])
            stack_transform = self.trans["stack"] if (
                "stack" in self.data_conf.test_trans) else None
            self.wave_test = WaveFormDataset(
                self.data_conf, data_type="test", transform=transform, stack_transform=stack_transform)

    def train_dataloader(self):
        return DataLoader(self.wave_train, batch_size=self.data_conf.train_batch_size, shuffle=self.data_conf.train_shuffle, num_workers=self.data_conf.num_workers)

    def val_dataloader(self):
        return DataLoader(self.wave_val, batch_size=self.data_conf.val_batch_size, shuffle=False, num_workers=self.data_conf.num_workers)

    def test_dataloader(self):
        return DataLoader(self.wave_test, batch_size=self.data_conf.test_batch_size, shuffle=False, num_workers=self.data_conf.num_workers)
