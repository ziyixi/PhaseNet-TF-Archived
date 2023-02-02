"""
load_conf.py

load configuration files for the project.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hydra.conf import HydraConf, JobConf, RunDir, SweepDir
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
    stack_ratio: float = 0.6  # ! hyper tune
    min_stack_gap: int = 100

    replace_noise: bool = True
    noise_replace_ratio: float = 0.05  # ! hyper tune

    scale_at_end: bool = True
    # * batch size and shuffle
    train_batch_size: int = 32
    val_batch_size: int = 1
    test_batch_size: int = 1
    train_shuffle: bool = True
    # * workers
    num_workers: int = 2

    # * PS freq picked by Fan, used for visualizing
    load_ps_freq: bool = True


@dataclass
class SpectrogramConfig:
    """
    Set confiuration to generate the spectrogram from the waveform dataset
    """
    n_fft: int = 256  # ! hyper tune
    hop_length: int = 1
    power: Optional[int] = 2
    window_fn: str = "hann"
    freqmin: float = 0.
    freqmax: float = 10.
    sampling_rate: int = 40
    height: int = 64
    width: int = 4800  # should equal to win_len*sampling_rate
    max_clamp: int = 3000  # ! hyper tune


@dataclass
class ModelConfig:
    """
    neural network model configuration
    """
    # can also be unet, deeplabv3+
    nn_model: str = "unet"
    in_channels: int = 3
    out_channels: int = 4
    init_features: int = 32  # ! hyper tune
    # n_freq is not used when train_with_spectrogram==False
    n_freq: int = 64  # should be the same as height in SpectrogramConfig
    first_layer_repeating_cnn: int = 3  # ! hyper tune

    encoder_conv_kernel_size: List[int] = field(
        default_factory=lambda: [5, 5])  # ! hyper tune
    decoder_conv_kernel_size: List[int] = field(
        default_factory=lambda: [5, 5])  # ! hyper tune

    encoder_decoder_depth: int = 5  # ! hyper tune

    # if False, train only use wave, should update other parameters by hand.
    train_with_spectrogram: bool = True

    # * below are deeplab configs, start with deeplab_
    # check when nn_model == deeplab
    deeplab_encoder_name: Optional[str] = "resnet34"
    deeplab_encoder_depth: Optional[int] = 5
    deeplab_encoder_weights: Optional[str] = None
    deeplab_encoder_output_stride: Optional[int] = 16
    deeplab_decoder_channels: Optional[int] = 256
    deeplab_decoder_atrous_rates: List[int] = field(default_factory=lambda: [
        12, 24, 36])
    deplab_upsampling: Optional[int] = 4


@dataclass
class TrainConfig:
    """
    the trainning configuration
    """
    # * random seed
    deterministic: bool = False
    use_random_seed: bool = True
    random_seed: int = 666
    # * basic configs
    learning_rate: float = 0.01  # ! hyper tune
    weight_decay: float = 1e-3  # ! hyper tune
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
    log_every_n_steps: int = 1
    # * loss func
    loss_func: str = "kl_div"
    # * when do seprate testing, load the ckpt path
    ckpt_path: Optional[str] = None
    # * run_type, whether train or hyper_tune
    run_type: str = "train"

    # * optimizer
    step_lr_milestones: List[int] = field(
        default_factory=lambda: [30, 60, 90, 120])
    step_lr_gamma: float = 0.5


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
    # save test to seprate folder
    log_test_seprate_folder: bool = False
    log_test_seprate_folder_path: str = ""
    # plot filtered waveform instead based on the PS/P arrival?
    # can be all (filter based on data config), P (P max range/fixed range), S (S max range)
    plot_waveform_based_on: str = "all"


@dataclass
class PostProcessConfig:
    """
    post process the model output, such as finding peak, cal metrics etc.
    """
    # metrics
    metrics_dt_threshold: float = 1.0

    # peaks
    sensitive_heights: Dict[str, float] = field(
        default_factory=lambda: {
            "TP": 0.5,
            "TS": 0.5,
            "TPS": 0.3
        })  # the peaks must have possibility at
    sensitive_distances: Dict[str, float] = field(
        default_factory=lambda: {
            "TP": 5.0,
            "TS": 5.0,
            "TPS": 5.0
        })  # when finding peaks, ignore close peaks in seconds
    # further analysis for test step
    save_test_step_to_disk: bool = False
    test_step_save_path: str = ""


@dataclass
class WandbConfig:
    """
    config wandb
    """
    job_name: str = "test"
    project_name: str = "PhaseNet-TF"
    model_log_freq: int = 200


@dataclass
class InferenceConfig:
    """
    config inference data and setup
    """
    sqlite_path: Path = MISSING  # the seed dir should be in the same dir as sqlite
    # the csv file indicating what data to handle
    continious_requirement_path: Path = MISSING
    # handle length in s (output as a single stream)
    continious_handle_time: int = 3600
    # sampling rate
    sampling_rate: int = 40
    # number of points in each window
    width: int = 4800
    # sliding windows step
    sliding_step: int = 2400

    # * trainner setting
    inference_batch_size: int = 1
    num_workers: int = 0

    # * MISC
    unit_is_m: bool = True

    # * checkpoint loading reference
    wandb_checkpoint_reference: str = MISSING


# * ======================================== * #
# * main conf
defaults = [
    {"data": "base_data"},
    {"spectrogram": "base_spectrogram"},
    {"model": "base_model"},
    {"train": "base_train"},
    {"visualize": "base_visualize"},
    {"postprocess": "base_postprocess"},
    {"wandb": "base_wandb"},
    {"inference": "base_inference"},
    "_self_"
]


@dataclass
class Hydra(HydraConf):
    # run: RunDir = RunDir("${output_dir}")
    run: RunDir = RunDir(
        dir="outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}")
    job: JobConf = JobConf(chdir=True)
    sweep: SweepDir = SweepDir(
        dir="outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}", subdir="${hydra.job.num}")


@dataclass
class Config:
    """
    the configuration for the project
    """
    defaults: List[Any] = field(default_factory=lambda: defaults)
    # * custom
    hydra: Hydra = Hydra()
    # * settings
    data: DataConfig = MISSING
    spectrogram: SpectrogramConfig = MISSING
    model: ModelConfig = MISSING
    train: TrainConfig = MISSING
    visualize: VisualizeConfig = MISSING
    postprocess: PostProcessConfig = MISSING
    wandb: WandbConfig = MISSING
    inference: InferenceConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="spectrogram", name="base_spectrogram", node=SpectrogramConfig)
cs.store(group="model", name="base_model", node=ModelConfig)
cs.store(group="train", name="base_train", node=TrainConfig)
cs.store(group="visualize", name="base_visualize", node=VisualizeConfig)
cs.store(group="postprocess", name="base_postprocess", node=PostProcessConfig)
cs.store(group="wandb", name="base_wandb", node=WandbConfig)
cs.store(group="inference", name="base_inference", node=InferenceConfig)
