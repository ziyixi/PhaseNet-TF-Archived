import logging
import warnings

import hydra

from phasenet.conf import Config
from phasenet.utils.helper import get_git_revision_short_hash
from pytorch_lightning import seed_everything
from phasenet.data.lighting_data import ContiniousSeedDataModule

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
    # * current version
    logger.info(f"current hash tag: {get_git_revision_short_hash()}")

    # * seed
    if cfg.train.use_random_seed:
        seed_everything(cfg.train.random_seed)

    light_data = ContiniousSeedDataModule(
        data_conf=cfg.data, inference_conf=cfg.inference)
    light_data.setup()
    for each in light_data.predict_dataloader():
        import torch
        print(each["net"], each["sta"], each["data"])
        print(torch.max(each["data"]), torch.min(each["data"]))
        torch.save(
            each["data"], "/mnt/home/xiziyi/Packages_Research/PhaseNet-PyTorch/notebook/check_inference_dataset/Z1.FONI.pt")
        break


if __name__ == "__main__":
    inference_app()
