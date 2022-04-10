import logging

import hydra
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from phasenet.conf.load_conf import Config
from phasenet.core.test import test_one_epoch
from phasenet.core.train import (criterion, get_optimizer, get_scheduler,
                                 train_one_epoch)
from phasenet.data.dataset import WaveFormDataset
from phasenet.data.transforms import (GenLabel, GenSgram, RandomShift,
                                      ScaleAmp, StackRand)
from phasenet.model.unet import UNet
from phasenet.utils.helper import get_git_revision_short_hash
from phasenet.utils.seed import setup_seed
from phasenet.utils.visualize import show_info_batch


@hydra.main(config_path="conf", config_name="config")
def train_app(cfg: Config) -> None:
    # * logger
    writer = SummaryWriter()
    log = logging.getLogger(__name__)
    log.info(f"current git hash {get_git_revision_short_hash()}")

    # * Set random number seed
    if cfg.train.use_random_seed:
        setup_seed(cfg.train.random_seed)
        log.info(f"using random seed {cfg.train.random_seed}")
    else:
        log.info(f"no random seed is used")

    # * locate device
    device = torch.device(cfg.train.device)
    log.info(f"using {device = }")

    # * load data
    trans = {
        "scale": ScaleAmp(cfg.preprocess),
        "shift": RandomShift(cfg.preprocess),
        "label": GenLabel(cfg.preprocess),
        "stack": StackRand(cfg.preprocess),
        "sgram": GenSgram(cfg.spectrogram, device=device)
    }
    data_train = WaveFormDataset(
        cfg, data_type=cfg.data.train_data_type, transform=Compose([trans[key] for key in cfg.preprocess.train_trans]), debug=cfg.data.data_debug, debug_dict={'size': cfg.data.data_debug_size})
    data_test = WaveFormDataset(
        cfg, data_type=cfg.data.test_data_type, transform=Compose([trans[key] for key in cfg.preprocess.test_trans]), debug=cfg.data.data_debug, debug_dict={'size': cfg.data.data_debug_size})
    if cfg.data.save_dataset:
        data_train.save(cfg.data.load_train)
        data_test.save(cfg.data.load_test)
    loader_train = DataLoader(
        data_train, batch_size=cfg.train.train_batch_size, shuffle=cfg.train.train_shuffle)
    loader_test = DataLoader(
        data_test, batch_size=cfg.train.test_batch_size, shuffle=False)

    # * show the target
    if cfg.visualize.save_target:
        show_info_batch(cfg, cfg.visualize.target_dir,
                        loader_train, example_num=cfg.visualize.example_num)

    # * train the model
    model = UNet(cfg)
    model.to(device)
    optimizer = get_optimizer(model.parameters(), cfg.train)
    main_lr_scheduler = get_scheduler(optimizer, len(loader_train), cfg.train)
    scaler = torch.cuda.amp.GradScaler() if cfg.train.use_amp else None
    for iepoch in range(cfg.train.epochs):
        log_predict = False if (
            iepoch != 0 and iepoch != cfg.train.epochs-1) else cfg.visualize.log_predict
        res = train_one_epoch(model, criterion, optimizer,
                              loader_train, main_lr_scheduler, use_amp=cfg.train.use_amp, device=device, log_predict=log_predict, scaler=scaler)
        res_test = test_one_epoch(model, criterion, loader_test,
                                  use_amp=cfg.train.use_amp, device=device)
        writer.add_scalar('Loss/train', res['loss_mean'], iepoch)
        writer.add_scalar('Loss/test', res_test['loss_mean'], iepoch)
        log.info(
            f"[#{iepoch}], train loss:{res['loss_mean']}, test loss:{res_test['loss_mean']}")
        # * show first epoch
        if iepoch == 0 and cfg.visualize.save_init and cfg.visualize.log_predict:
            show_info_batch(cfg, cfg.visualize.init_dir, loader_train,
                            predict=res['predict'], example_num=cfg.visualize.example_num)

    # * show the final plot
    if cfg.visualize.save_final and cfg.visualize.log_predict:
        show_info_batch(cfg, cfg.visualize.final_dir,
                        loader_train, predict=res['predict'], example_num=cfg.visualize.example_num)

    # * exist log
    writer.close()


if __name__ == "__main__":
    train_app()
