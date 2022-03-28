import random
from os.path import join

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
from phasenet.utils.visualize import show_info


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
    loader_train = DataLoader(data_train, batch_size=2, shuffle=False)
    # * show the initial input
    sampling_rate = cfg.spectrogram.sampling_rate
    base_save_dir = "/Users/ziyixi/OneDrive - Michigan State University/Packages_Research/PhaseNet-PyTorch/figs"

    target_save_dir = join(base_save_dir, "test_train_simple", "target")
    for each_batch in loader_train:
        show_info(each_batch, phases=cfg.data.phases,  save_dir=target_save_dir, sampling_rate=sampling_rate, x_range=[
            0, cfg.preprocess.win_length], freq_range=[cfg.spectrogram.freqmin, cfg.spectrogram.freqmax], progress=False, global_max=False)

    # * test batch and plot
    model = UNet(cfg)
    optimizer = get_optimizer(model.parameters(), cfg.train)
    main_lr_scheduler = get_scheduler(optimizer, len(loader_train), cfg.train)
    for iepoch in range(cfg.train.epochs):
        res = train_one_epoch(model, criterion, optimizer,
                              loader_train, main_lr_scheduler, log=True)
        print(f"{iepoch =},{res['loss_mean'] =},{res['loss'] =}")
        # * show first epoch
        if iepoch == 0:
            init_save_dir = join(base_save_dir, "test_train_simple", "init")
            for ibatch, each_batch in enumerate(loader_train):
                show_info(each_batch, phases=cfg.data.phases,  save_dir=init_save_dir, sampling_rate=sampling_rate, x_range=[
                    0, cfg.preprocess.win_length], freq_range=[cfg.spectrogram.freqmin, cfg.spectrogram.freqmax], progress=False, global_max=False, predict=res['predict'][ibatch])

    # * show the final plot
    final_save_dir = join(base_save_dir, "test_train_simple", "final")
    for ibatch, each_batch in enumerate(loader_train):
        show_info(each_batch, phases=cfg.data.phases,  save_dir=final_save_dir, sampling_rate=sampling_rate, x_range=[
            0, cfg.preprocess.win_length], freq_range=[cfg.spectrogram.freqmin, cfg.spectrogram.freqmax], progress=False, global_max=False, predict=res['predict'][ibatch])


if __name__ == "__main__":
    train_app()
