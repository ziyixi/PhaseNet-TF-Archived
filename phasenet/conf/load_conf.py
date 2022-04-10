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
    load_train: str = MISSING
    load_test: str = MISSING
    phases: List[str] = MISSING
    data_debug: bool = False
    data_debug_size: int = 8
    save_dataset: bool = False
    train_data_type: str = "load_train"
    test_data_type: str = "load_test"


@dataclass
class PreprocessConfig:
    """
    preprocess for the dataset and dataloader
    """
    # cut win in dataset
    win_length: float = 120.
    left_extend: float = 10.
    right_extend: float = 110.
    # label
    label_shape: str = "gaussian"
    label_width: int = 120
    # signal processing
    taper_percentage: float = 0.05
    filter_freqmin: float = 1
    filter_freqmax: float = 10.
    filter_corners: int = 4
    filter_zerophase: bool = False
    # random shift
    width: int = 4800
    # stack
    stack_ratio: float = 0.6
    min_stack_gap: int = 100
    # scale
    scale_max_amp: float = 1.0
    scale_global_max: bool = True
    # train transforms
    train_trans: List[str] = field(default_factory=lambda: [
                                   "scale", "shift", "label", "stack", "sgram"])
    test_trans: List[str] = field(default_factory=lambda: [
        "scale", "shift", "label", "sgram"])


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
    max_clamp: int = 50


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
    use_random_seed: bool = True
    random_seed: int = 666
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 20
    lr_warmup_epochs: int = 0
    device: str = "cpu"
    train_batch_size: int = 32
    test_batch_size: int = 32
    train_shuffle: bool = True
    use_amp: bool = True


@dataclass
class ProfileConfig:
    """
    the profiling configuration
    """
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 2


@dataclass
class VisualizeConfig:
    """
    the visualization configuration
    """
    fig_dir: str = MISSING
    target_dir: str = MISSING
    init_dir: str = MISSING
    final_dir: str = MISSING
    save_target: bool = False
    save_init: bool = False
    save_final: bool = False
    log_predict: bool = False
    example_num: int = 8


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
    profile: ProfileConfig = MISSING
    visualize: VisualizeConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="preprocess", name="base_preprocess", node=PreprocessConfig)
cs.store(group="spectrogram", name="base_spectrogram", node=SpectrogramConfig)
cs.store(group="model", name="base_model", node=ModelConfig)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="profile", name="base_profile", node=ProfileConfig)
cs.store(group="visualize", name="base_visualize", node=VisualizeConfig)
