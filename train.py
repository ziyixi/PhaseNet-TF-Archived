import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from phasenet.conf.load_conf import Config
from phasenet.data.dataset import WaveFormDataset
from phasenet.data.transforms import GenLabel, GenSgram, ScaleAmp
from phasenet.model.unet import UNet
import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set random number seed
setup_seed(20)


@hydra.main(config_path="conf", config_name="config")
def train_app(cfg: Config) -> None:
    # * load data
    trans_label = GenLabel(
        label_shape=cfg.preprocess.label_shape, label_width=cfg.preprocess.label_width)
    trans_scale = ScaleAmp(max_amp=1, global_max=True)
    trans_sgram = GenSgram(n_fft=cfg.spectrogram.n_fft, hop_length=cfg.spectrogram.hop_length, power=cfg.spectrogram.power, window_fn=cfg.spectrogram.window_fn,
                           freqmin=cfg.spectrogram.freqmin, freqmax=cfg.spectrogram.freqmax, sampling_rate=cfg.spectrogram.sampling_rate, height=cfg.spectrogram.height, width=cfg.spectrogram.width)
    composed = Compose([trans_label, trans_scale, trans_sgram])

    data_train = WaveFormDataset(
        cfg, data_type="train", transform=composed, progress=False, debug=True, debug_dict={'size': 4})
    loader_train = DataLoader(data_train, batch_size=2, shuffle=True)
    # * test batch and plot
    first_batch = next(iter(loader_train))
    print("----input info----")
    print(first_batch.keys())
    print(first_batch['sgram'].shape)
    print(first_batch['data'].shape)
    print(first_batch['label'].shape)

    print("----in progress info----")
    model = UNet(cfg)
    first_batch_res = model(first_batch['sgram'])

    print("----output info----")
    print(first_batch_res['predict'].shape)


if __name__ == "__main__":
    train_app()
