import logging
import warnings
from pathlib import Path

import hydra
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelSummary

from phasenet.conf import Config
from phasenet.core.lighting_model import PhaseNetModel
from phasenet.data.lighting_data import ContiniousSeedDataModule
from phasenet.model.segmentation_models import create_smp_model
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
warnings.filterwarnings(
    "ignore", ".*SELECT statement has a cartesian product*")


@hydra.main(config_path=".", config_name="base_config", version_base="1.2")
def inference_app(cfg: Config) -> None:
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
        SegModel = create_smp_model(model_conf=cfg.model)
        light_model = PhaseNetModel(SegModel, cfg)

    light_data = ContiniousSeedDataModule(
        data_conf=cfg.data, inference_conf=cfg.inference)

    # * callbacks
    callbacks = []
    callbacks.append(ModelSummary(max_depth=2))

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
        deterministic=train_conf.deterministic,
        devices=(
            train_conf.distributed_devices if train_conf.accelerator == "gpu" else None),
        precision=precision,
        strategy=(train_conf.strategy if train_conf.accelerator ==
                  "gpu" else None),
        sync_batchnorm=train_conf.sync_batchnorm,
        num_sanity_val_steps=0,  # no need to do this check outside development
    )

    # * download the checkpoint from wandb
    run = wandb.init(project=cfg.wandb.project_name)
    artifact = run.use_artifact(
        cfg.inference.wandb_checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    logger.info(f"checkpoint is downloaded to {artifact_dir}")

    trainer.predict(model=light_model, dataloaders=light_data.predict_dataloader(),
                    ckpt_path="/mnt/home/xiziyi/Packages_Research/PhaseNet-PyTorch/outputs/baseline/2023-01-27_17-10-42/0/PhaseNet-TF/b2zisxi6/checkpoints/epoch=66-loss_val=143.78.ckpt")


if __name__ == "__main__":
    inference_app()
