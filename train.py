import warnings

import hydra
from pytorch_lightning import Trainer, seed_everything

from phasenet.conf.load_conf import Config
from phasenet.core.lighting_model import PhaseNetModel
from phasenet.data.lighting_data import WaveFormDataModule
from phasenet.model.unet import UNet

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*")


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
    # * prepare trainner
    trainer = Trainer(
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
    )
    # * train and val
    light_data.setup(stage="fit")
    trainer.fit(light_model, light_data)
    # * test
    light_data.setup(stage="test")
    trainer.test(datamodule=light_data, ckpt_path='best')


if __name__ == "__main__":
    train_app()
