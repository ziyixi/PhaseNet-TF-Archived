import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from phasenet.conf.load_conf import Config
from phasenet.data.dataset import WaveFormDataset
from phasenet.data.transforms import GenLabel, GenSgram, ScaleAmp
from phasenet.model.unet import UNet
from phasenet.utils.visualize import show_data_batch, show_sgram, show_input
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

    # data_train = WaveFormDataset(
    # cfg, data_type="train", transform=composed, progress=False, debug=True, debug_dict={'size': 20})
    data_train = WaveFormDataset(
        cfg, data_type="train", transform=composed, progress=True, debug=False)
    loader_train = DataLoader(data_train, batch_size=20, shuffle=True)
    # * test batch and plot
    first_batch = next(iter(loader_train))
    sampling_rate = cfg.spectrogram.sampling_rate
    save_dir = "/Users/ziyixi/OneDrive - Michigan State University/Packages_Research/PhaseNet-PyTorch/figs"
    show_input(first_batch, phases=cfg.data.phases,  save_dir=save_dir, sampling_rate=sampling_rate, x_range=[
               0, cfg.preprocess.win_length], freq_range=[cfg.spectrogram.freqmin, cfg.spectrogram.freqmax], progress=True, global_max=True)
    # save
    # np.save('/Users/ziyixi/OneDrive - Michigan State University/Packages_Research/PhaseNet-PyTorch/jupyter/test.npy',
    #         first_batch['data'])
    # model
    # model = UNet(cfg)
    # test_batch: torch.Tensor = first_batch['data']
    # res = model(test_batch)
    # print(res['out'].shape)
    # # plot
    # save_dir = "/Users/ziyixi/OneDrive - Michigan State University/Packages_Research/PhaseNet-PyTorch/figs"
    # sampling_rate = 40
    # show_data_batch(first_batch, phases=cfg.data.phases,
    #                 save_dir=save_dir, sampling_rate=sampling_rate, show_output=True, output_batch=res['out'].squeeze(3))
    # show_sgram(res['sgram'], first_batch['key'], save_dir)


if __name__ == "__main__":
    train_app()
