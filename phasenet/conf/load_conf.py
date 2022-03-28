"""
load_conf.py

load configuration files for the project.
"""
from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataConfig:
    """
    the dataset path configuration 
    """
    train: str = MISSING
    test: str = MISSING
    phases: List[str] = MISSING


@dataclass
class PreprocessConfig:
    """
    preprocess for the dataset and dataloader
    """
    win_length: float = 120.
    left_extend: float = 10.
    right_extend: float = 110.
    label_shape: str = "gaussian"
    label_width: int = 120
    # signal processing
    taper_percentage: float = 0.05
    filter_freqmin: float = 1
    filter_freqmax: float = 10.
    filter_corners: int = 4
    filter_zerophase: bool = False


@dataclass
class SpectrogramConfig:
    """
    Set confiuration to generate the spectrogram from the waveform dataset
    """
    n_fft: int = 128
    hop_length: int = 1
    power: int = 2
    window_fn: str = "hann"
    freqmin: float = 1
    freqmax: float = 10.
    sampling_rate: int = 40
    height: int = 64
    width: int = 4800  # should equal to win_len*sampling_rate


@dataclass
class ModelConfig:
    """
    neural network model configuration
    """
    nn_model: str = "unet"
    in_channels: int = 3
    out_channels: int = 4
    init_features: int = 32
    n_freq: int = 64

    encoder_conv_kernel_size: List[int] = field(default_factory=lambda: [3, 3])
    encoder_pool_kernel_size: List[int] = field(default_factory=lambda: [2, 2])
    encoder_pool_stride: List[int] = field(default_factory=lambda: [2, 2])

    decoder_conv_kernel_size: List[int] = field(default_factory=lambda: [3, 3])
    decoder_pool_kernel_size: List[int] = field(default_factory=lambda: [2, 2])
    decoder_pool_stride: List[int] = field(default_factory=lambda: [2, 2])


@dataclass
class TrainConfig:
    """
    teh trainning configuration
    """
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 20
    lr_warmup_epochs: int = 0


@dataclass
class Config:
    """
    the configuration for the project
    """
    data: DataConfig = MISSING
    preprocess: PreprocessConfig = MISSING
    spectrogram: SpectrogramConfig = MISSING
    model: ModelConfig = MISSING
    train: TrainConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="preprocess", name="base_preprocess", node=PreprocessConfig)
cs.store(group="spectrogram", name="base_spectrogram", node=SpectrogramConfig)
cs.store(group="model", name="base_model", node=ModelConfig)
cs.store(group="train", name="base_train", node=TrainConfig)
