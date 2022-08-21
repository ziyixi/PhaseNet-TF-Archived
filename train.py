import logging
import warnings

import hydra
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, ModelSummary)

from phasenet.conf import Config
from phasenet.core.lighting_model import PhaseNetModel
from phasenet.data.lighting_data import WaveFormDataModule
from phasenet.model.unet import UNet
from phasenet.utils.helper import get_git_revision_short_hash

logger = logging.getLogger('lightning')

# * ignores
warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings(
    "ignore", ".*Failure to do this will result in PyTorch skipping the first value*")
warnings.filterwarnings(
    "ignore", ".*During `trainer.test()`, it is recommended to use `Trainer(devices=1)`*")


@hydra.main(config_path=".", config_name="base_config", version_base="1.2")
def train_app(cfg: Config) -> None:
    train_conf = cfg.train
    # * current version
    logger.info(f"current hash tag: {get_git_revision_short_hash()}")

    # * seed
    if train_conf.use_random_seed:
        seed_everything(train_conf.random_seed)

    # * prepare light data and model
    if cfg.model.nn_model == "unet":
        light_model = PhaseNetModel(UNet, cfg)
    else:
        raise Exception(f"model {cfg.model.nn_model} is not supported.")
    light_data = WaveFormDataModule(cfg.data)
    light_data.prepare_data()

    # * callbacks
    callbacks = []
    callbacks.append(LearningRateMonitor(
        logging_interval='epoch'))
    callbacks.append(ModelSummary(max_depth=2))
    callbacks.append(ModelCheckpoint(
        save_top_k=3, monitor="loss_val", mode="min", filename="{epoch:02d}-{loss_val:.2f}"))
    callbacks.append(EarlyStopping(monitor="loss_val",
                     patience=10, verbose=False, mode="min"))

    # * prepare trainner
    precision = 32
    if train_conf.use_amp:
        if train_conf.use_a100:
            precision = "bf16"
        else:
            precision = 16

    trainer = Trainer(
        callbacks=callbacks,
        accelerator=train_conf.accelerator,
        deterministic=train_conf.use_random_seed,
        devices=(
            train_conf.distributed_devices if train_conf.accelerator == "gpu" else None),
        precision=precision,
        max_epochs=train_conf.epochs,
        strategy=(train_conf.strategy if train_conf.accelerator ==
                  "gpu" else None),
        limit_train_batches=(
            train_conf.limit_train_batches if train_conf.limit_train_batches else None),
        limit_val_batches=(
            train_conf.limit_val_batches if train_conf.limit_val_batches else None),
        limit_test_batches=(
            train_conf.limit_test_batches if train_conf.limit_test_batches else None),
        log_every_n_steps=train_conf.log_every_n_steps,
        sync_batchnorm=train_conf.sync_batchnorm,
        num_sanity_val_steps=0,  # no need to do this check outside development
    )

    # * train and val
    light_data.setup(stage="fit")
    trainer.fit(light_model, light_data)
    # * test

    light_data.setup(stage="test")
    trainer.test(datamodule=light_data, ckpt_path='best')


if __name__ == "__main__":
    train_app()
