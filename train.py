import random

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from phasenet.conf.load_conf import Config
from phasenet.core.train import (criterion, get_optimizer, get_scheduler,
                                 train_one_epoch)
from phasenet.data.dataset import WaveFormDataset
from phasenet.data.transforms import GenLabel, GenSgram, ScaleAmp
from phasenet.model.unet import UNet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set random number seed
setup_seed(20)


@hydra.main(config_path="conf", config_name="config")
def train_app(cfg: Config) -> None:
    # * load data
    trans_label = GenLabel(
        label_shape=cfg.preprocess.label_shape, label_width=cfg.preprocess.label_width)
    trans_scale = ScaleAmp(max_amp=1, global_max=True)
    trans_sgram = GenSgram(n_fft=cfg.spectrogram.n_fft, hop_length=cfg.spectrogram.hop_length, power=cfg.spectrogram.power, window_fn=cfg.spectrogram.window_fn,
                           freqmin=cfg.spectrogram.freqmin, freqmax=cfg.spectrogram.freqmax, sampling_rate=cfg.spectrogram.sampling_rate, height=cfg.spectrogram.height, width=cfg.spectrogram.width)
    composed = Compose([trans_label, trans_scale, trans_sgram])

    data_train = WaveFormDataset(
        cfg, data_type="train", transform=composed, progress=False, debug=True, debug_dict={'size': 8})
    loader_train = DataLoader(data_train, batch_size=2, shuffle=True)
    # * test batch and plot
    model = UNet(cfg)
    optimizer = get_optimizer(model.parameters(), cfg.train)
    main_lr_scheduler = get_scheduler(optimizer, len(loader_train), cfg.train)
    for iepoch in range(cfg.train.epochs):
        train_one_epoch(model, criterion, optimizer,
                        loader_train, main_lr_scheduler)


if __name__ == "__main__":
    train_app()
