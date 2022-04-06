import logging
from os.path import join

import hydra
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from phasenet.conf.load_conf import Config
from phasenet.core.train import (criterion, get_optimizer, get_scheduler,
                                 train_one_epoch)
from phasenet.core.test import test_one_epoch
from phasenet.data.dataset import WaveFormDataset
from phasenet.data.transforms import GenLabel, GenSgram, RandomShift, ScaleAmp, StackRand
from phasenet.model.unet import UNet
from phasenet.utils.seed import setup_seed
from phasenet.utils.visualize import show_info_batch

# * custome settings
DEVICE = 'cuda:2'
FIG_DIR = "/Users/ziyixi/Library/CloudStorage/OneDrive-MichiganStateUniversity/Packages_Research/PhaseNet-PyTorch/figs"
# * logger
log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def train_app(cfg: Config) -> None:
    writer = SummaryWriter()

    # * Set random number seed
    setup_seed(20)

    # * locate device
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    log.info(f"using {device = }")

    # * load data
    trans_scale = ScaleAmp(max_amp=1, global_max=True)
    trans_shift = RandomShift(width=cfg.preprocess.width)
    trans_label = GenLabel(
        label_shape=cfg.preprocess.label_shape, label_width=cfg.preprocess.label_width)
    trans_stack = StackRand(stack_ratio=cfg.preprocess.stack_ratio,
                            min_stack_gap=cfg.preprocess.min_stack_gap)
    trans_sgram = GenSgram(n_fft=cfg.spectrogram.n_fft, hop_length=cfg.spectrogram.hop_length, power=cfg.spectrogram.power, window_fn=cfg.spectrogram.window_fn,
                           freqmin=cfg.spectrogram.freqmin, freqmax=cfg.spectrogram.freqmax, sampling_rate=cfg.spectrogram.sampling_rate,
                           height=cfg.spectrogram.height, width=cfg.spectrogram.width, max_clamp=cfg.spectrogram.max_clamp, device=device)

    data_train = WaveFormDataset(
        cfg, data_type="load_train", transform=Compose([trans_scale, trans_shift, trans_label, trans_stack, trans_sgram]), progress=True, debug=True, debug_dict={'size': 8})
    data_test = WaveFormDataset(
        cfg, data_type="load_test", transform=Compose([trans_scale, trans_shift, trans_label, trans_stack, trans_sgram]), progress=True, debug=True, debug_dict={'size': 8})
    # data_train.save(cfg.data.load_train)
    loader_train = DataLoader(data_train, batch_size=2, shuffle=False)
    loader_test = DataLoader(data_test, batch_size=2, shuffle=False)

    # * show the initial input
    target_save_dir = join(FIG_DIR, "test_train_simple", "target")
    show_info_batch(cfg, target_save_dir, loader_train)

    # * train the model
    model = UNet(cfg)
    model.to(device)
    optimizer = get_optimizer(model.parameters(), cfg.train)
    main_lr_scheduler = get_scheduler(optimizer, len(loader_train), cfg.train)
    for iepoch in range(cfg.train.epochs):
        res = train_one_epoch(model, criterion, optimizer,
                              loader_train, main_lr_scheduler, device=device, enable_log=True)
        res_test = test_one_epoch(model, criterion, loader_test, device=device)
        writer.add_scalar('Loss/train', res['loss_mean'], iepoch)
        writer.add_scalar('Loss/test', res_test['loss_mean'], iepoch)
        log.info(
            f"[#{iepoch}], train loss:{res['loss_mean']}, test loss:{res_test['loss_mean']}")
        # * show first epoch
        if iepoch == 0:
            init_save_dir = join(FIG_DIR, "test_train_simple", "init")
            show_info_batch(cfg, init_save_dir, loader_train,
                            predict=res['predict'])

    # * show the final plot
    final_save_dir = join(FIG_DIR, "test_train_simple", "final")
    show_info_batch(cfg, final_save_dir, loader_train, predict=res['predict'])

    writer.close()


if __name__ == "__main__":
    train_app()
