import logging
import warnings

import hydra
from pytorch_lightning import Trainer, seed_everything

from phasenet.conf import Config
from phasenet.core.lighting_model import MeanStdEstimator
from phasenet.data.lighting_data import WaveFormDataModule
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
def test_app(cfg: Config) -> None:
    train_conf = cfg.train
    # * current version
    logger.info(f"current hash tag: {get_git_revision_short_hash()}")

    # * seed
    if train_conf.use_random_seed:
        seed_everything(train_conf.random_seed)

    # * prepare light data and model
    light_model = MeanStdEstimator(cfg)

    light_data = WaveFormDataModule(
        data_conf=cfg.data, run_type=cfg.train.run_type)
    light_data.prepare_data()

    # * prepare trainner
    precision = 32
    if train_conf.use_amp:
        if train_conf.use_a100:
            precision = "bf16"
        else:
            precision = 16

    trainer = Trainer(
        accelerator=train_conf.accelerator,
        deterministic=train_conf.use_random_seed,
        devices=(
            [train_conf.distributed_devices[0]] if train_conf.accelerator == "gpu" else None),
        precision=precision,
        max_epochs=1,
        strategy="dp",
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


if __name__ == "__main__":
    test_app()
