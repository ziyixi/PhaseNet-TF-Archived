"""
load_conf.py

load configuration files for the project.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataConfig:
    """
    the dataset path configuration 
    """
    # * data path
    data_dir: str = MISSING
    train: str = MISSING
    test: str = MISSING
    val: str = MISSING
    phases: List[str] = field(default_factory=lambda: ["TP", "TS", "TPS"])
    # * cut win in dataset
    win_length: float = 120.
    left_extend: float = 10.
    right_extend: float = 110.
    avoid_first_ten_seconds: bool = True  # avoid taper effect
    avoid_last_ten_seconds: bool = True  # avoid taper effect
    width: int = 4800
    # * label
    label_shape: str = "gaussian"
    label_width: int = 120
    # * signal processing
    filter_freqmin: float = 1
    filter_freqmax: float = 10.
    filter_corners: int = 4
    filter_zerophase: bool = False
    # * transforms, data agument
    train_trans: List[str] = field(default_factory=lambda: [
        "shift", "scale", "label"])
    val_trans: List[str] = field(default_factory=lambda: [
        "shift", "scale", "label"])
    test_trans: List[str] = field(default_factory=lambda: [
        "shift", "scale", "label"])

    stack: bool = True
    stack_ratio: float = 0.6
    min_stack_gap: int = 100

    replace_noise: bool = True
    noise_replace_ratio: float = 0.05

    scale_at_end: bool = True
    scale_max_amp: float = 1.0
    scale_global_max: bool = True
    scale_norm: bool = True  # normalization to std distribution, ignore scale_max_amp
    # * batch size and shuffle
    train_batch_size: int = 32
    val_batch_size: int = 1
    test_batch_size: int = 1
    train_shuffle: bool = True
    # * workers
    num_workers: int = 2


@dataclass
class SpectrogramConfig:
    """
    Set confiuration to generate the spectrogram from the waveform dataset
    """
    n_fft: int = 256
    hop_length: int = 1
    power: int = 2
    window_fn: str = "hann"
    freqmin: float = 0.
    freqmax: float = 10.
    sampling_rate: int = 40
    height: int = 64
    width: int = 4800  # should equal to win_len*sampling_rate
    max_clamp: int = 3000


@dataclass
class ModelConfig:
    """
    neural network model configuration
    """
    nn_model: str = "unet"
    in_channels: int = 3
    out_channels: int = 4
    init_features: int = 32
    n_freq: int = 64  # should be the same as height in SpectrogramConfig
    first_layer_repeating_cnn: int = 3

    encoder_conv_kernel_size: List[int] = field(default_factory=lambda: [5, 5])
    decoder_conv_kernel_size: List[int] = field(default_factory=lambda: [5, 5])


@dataclass
class TrainConfig:
    """
    the trainning configuration
    """
    # * random seed
    use_random_seed: bool = True
    random_seed: int = 666
    # * basic configs
    learning_rate: float = 0.01
    weight_decay: float = 1e-3
    epochs: int = 100
    sync_batchnorm: bool = True
    # * acceleration
    accelerator: str = "cpu"
    strategy: Optional[str] = None
    use_amp: bool = False
    use_a100: bool = False
    distributed_devices: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3])
    # * test on local
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    limit_test_batches: Optional[int] = None
    # * logging
    log_every_n_steps: int = 20


@dataclass
class VisualizeConfig:
    """
    the visualization configuration
    """
    example_num: int = 8
    # considered first, if so, will log final and also consider log_epoch
    log_train: bool = False
    log_val: bool = False
    log_test: bool = False
    # if log every several epochs
    log_epoch: Optional[int] = None
    # figs
    sgram_threshold: Optional[int] = None


@dataclass
class Config:
    """
    the configuration for the project
    """
    data: DataConfig = MISSING
    spectrogram: SpectrogramConfig = MISSING
    model: ModelConfig = MISSING
    train: TrainConfig = MISSING
    visualize: VisualizeConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="spectrogram", name="base_spectrogram", node=SpectrogramConfig)
cs.store(group="model", name="base_model", node=ModelConfig)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="visualize", name="base_visualize", node=VisualizeConfig)
