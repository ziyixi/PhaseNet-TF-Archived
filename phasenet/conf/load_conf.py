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


# @dataclass
# class SpectrogramConfig:
#     """
#     set spectrogram conversion for the input waveform dataset
#     """
#     n_fft: int = 128  # width of each FFT window (number of frequency bins is n_fft//2 + 1)
#     hop_length: int = 8  # interval between consecutive windows
#     window_fn: str = "hann"  # windowing function
#     # if true, apply the function f(x) = log(1 + x) pointwise to the output of the spectrogram
#     log_transform: bool = True
#     # if true, return the magnitude of the complex value in each time-frequency bin
#     magnitude: bool = False
#     phase: bool = False
#     # if true, allow gradients to propagate through the spectrogram transformation
#     grad: bool = False
#     # if true, remove the zero frequency row from the spectrogram
#     discard_zero_freq: bool = False
#     select_freq: bool = True
#     dt: float = 0.025
#     fmin: float = 1.
#     fmax: float = 10.

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
    n_freq: int = 57
    use_stft: bool = True
    encoder_kernel_size: List[int] = field(default_factory=lambda: [2, 3])
    decoder_kernel_size: List[int] = field(default_factory=lambda: [2, 3])
    encoder_stride: List[int] = field(default_factory=lambda: [2, 2])
    decoder_stride: List[int] = field(default_factory=lambda: [2, 2])
    encoder_padding: List[int] = field(default_factory=lambda: [0, 1])
    decoder_padding: List[int] = field(default_factory=lambda: [0, 1])


@dataclass
class Config:
    """
    the configuration for the project
    """
    data: DataConfig = MISSING
    preprocess: PreprocessConfig = MISSING
    spectrogram: SpectrogramConfig = MISSING
    model: ModelConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="preprocess", name="base_preprocess", node=PreprocessConfig)
cs.store(group="spectrogram", name="base_spectrogram", node=SpectrogramConfig)
cs.store(group="model", name="base_model", node=ModelConfig)
