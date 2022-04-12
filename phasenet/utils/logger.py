from omegaconf import OmegaConf
import logging


def get_logger(name=None):
    # refer to https://github.com/facebookresearch/hydra/issues/1126#issuecomment-724331080
    hydra_conf = OmegaConf.load('.hydra/hydra.yaml')
    logging.config.dictConfig(OmegaConf.to_container(
        hydra_conf.hydra.job_logging, resolve=True))
    return logging.getLogger(name)
