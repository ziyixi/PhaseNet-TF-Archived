import logging

import hydra
import torch
import torch.multiprocessing as mp
from torch.distributed import reduce, ReduceOp
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
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
from phasenet.utils.distribute import cleanup_distribute, setup_distribute
from phasenet.utils.helper import get_git_revision_short_hash
from phasenet.utils.seed import setup_seed
from phasenet.utils.visualize import show_info_batch


@hydra.main(config_path="conf", config_name="config")
def train_app(cfg: Config) -> None:
    # * logger
    # see discussion https://github.com/facebookresearch/hydra/issues/1126
    # we have to put it outside the spawned processes
    log = logging.getLogger(__name__)
    writer = SummaryWriter()
    log.info(f"current git hash {get_git_revision_short_hash()}")
    # * spawn
    if cfg.train.distributed:
        mp.spawn(train_app_distribute,
                 args=(cfg, log, writer),
                 nprocs=len(cfg.train.distributed_devices),
                 join=True)
    else:
        train_app_distribute(0, cfg, log, writer)


def train_app_distribute(rank: int, cfg: Config, log: logging.Logger, writer: SummaryWriter):
    # * spawn
    if cfg.train.distributed:
        setup_distribute(rank, len(cfg.train.distributed_devices),
                         cfg.train.distributed_master_port)

    # * Set random number seed
    if cfg.train.use_random_seed:
        setup_seed(cfg.train.random_seed)
        if rank == 0:
            log.info(f"using random seed {cfg.train.random_seed}")
    else:
        if rank == 0:
            log.info(f"no random seed is used")

    # * locate device
    if cfg.train.distributed:
        gpu_idx = cfg.train.distributed_devices[rank]
        device = torch.device(f"cuda:{gpu_idx}")
        if rank == 0:
            log.info(f"using devices {cfg.train.distributed_devices}")
    else:
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
    if cfg.data.save_dataset and rank == 0:
        data_train.save(cfg.data.load_train)
        data_test.save(cfg.data.load_test)

    if cfg.train.distributed:
        train_sampler = DistributedSampler(data_train)
        test_sampler = DistributedSampler(data_test)
    else:
        train_sampler = RandomSampler(
            data_train) if cfg.train.train_shuffle else SequentialSampler(data_train)
        test_sampler = SequentialSampler(data_test)

    loader_train = DataLoader(
        data_train, batch_size=cfg.train.train_batch_size, sampler=train_sampler, drop_last=True)
    loader_test = DataLoader(
        data_test, batch_size=cfg.train.test_batch_size, sampler=test_sampler)

    # * show the target
    if cfg.visualize.save_target:
        show_info_batch(cfg, cfg.visualize.target_dir,
                        loader_train, example_num=cfg.visualize.example_num)

    # * train the model
    model = UNet(cfg)
    model.to(device)
    if cfg.train.distributed:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device)

    optimizer = get_optimizer(model.parameters(), cfg.train)
    main_lr_scheduler = get_scheduler(optimizer, len(loader_train), cfg.train)
    scaler = torch.cuda.amp.GradScaler() if cfg.train.use_amp else None
    for iepoch in range(cfg.train.epochs):
        if cfg.train.distributed:
            train_sampler.set_epoch(iepoch)

        log_predict = False if (
            iepoch != 0 and iepoch != cfg.train.epochs-1) else cfg.visualize.log_predict
        res = train_one_epoch(model, criterion, optimizer,
                              loader_train, main_lr_scheduler, use_amp=cfg.train.use_amp, device=device, log_predict=log_predict, scaler=scaler)
        res_test = test_one_epoch(model, criterion, loader_test,
                                  use_amp=cfg.train.use_amp, device=device)
        train_loss_mean = res['loss_mean']
        test_loss_mean = res_test['loss_mean']
        if cfg.train.distributed:
            reduce(train_loss_mean, 0, ReduceOp.SUM)
            reduce(test_loss_mean, 0, ReduceOp.SUM)
            if rank == 0:
                train_loss_mean /= len(cfg.train.distributed_devices)
                test_loss_mean /= len(cfg.train.distributed_devices)

        if rank == 0:
            writer.add_scalar('Loss/train', train_loss_mean.item(), iepoch)
            writer.add_scalar('Loss/test', test_loss_mean.item(), iepoch)
            log.info(
                f"[#{iepoch}], train loss:{train_loss_mean.item()}, test loss:{test_loss_mean.item()}")
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

    # * spawn
    if cfg.train.distributed:
        cleanup_distribute()


if __name__ == "__main__":
    train_app()
