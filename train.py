import warnings

import hydra
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from phasenet.conf.load_conf import Config
from phasenet.core.lighting_model import PhaseNetModel
from phasenet.data.lighting_data import WaveFormDataModule
from phasenet.model.unet import UNet

# * ignores
warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings(
    "ignore", ".*Failure to do this will result in PyTorch skipping the first value*")
warnings.filterwarnings(
    "ignore", ".*During `trainer.test()`, it is recommended to use `Trainer(devices=1)`*")


@hydra.main(config_path="conf", config_name="config")
def train_app(cfg: Config) -> None:
    train_conf = cfg.train
    # * seed
    if train_conf.use_random_seed:
        seed_everything(train_conf.random_seed)
    # * prepare light data and model
    light_model = PhaseNetModel(UNet, cfg)
    light_data = WaveFormDataModule(cfg.data)
    light_data.prepare_data()
    # * callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')  # monitor lr
    # * prepare trainner
    trainer = Trainer(
        callbacks=[lr_monitor],
        accelerator=train_conf.accelerator,
        deterministic=train_conf.use_random_seed,
        devices=(
            train_conf.distributed_devices if train_conf.accelerator == "gpu" else None),
        precision=(16 if train_conf.use_amp else 32),
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
