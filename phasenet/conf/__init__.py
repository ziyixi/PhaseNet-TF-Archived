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
    data_dir: str = MISSING  # the directory containing train/test/val h5 files
    train: str = MISSING  # the training h5 file basename
    test: str = MISSING  # the test h5 file basename
    val: str = MISSING  # the val h5 file basename
    # The phases to predict, h5 file should contain the phases' info
    phases: List[str] = field(default_factory=lambda: ["TP", "TS", "TPS"])
    # * cut win in dataset
    win_length: float = 120.  # the window length in seconds
    left_extend: float = 10.  # seconds before the first P arrival in the window
    right_extend: float = 110.  # seconds after the first P arrival in the window
    width: int = 4800  # the actual window length in # if points
    # * label
    label_shape: str = "gaussian"  # the label shape, can be gaussian or triangle
    label_width: int = 120  # the full width of the label in # of points
    # * signal processing
    filter_freqmin: float = 1  # the filter min freq
    filter_freqmax: float = 10.  # the filter max freq
    filter_corners: int = 4  # the filter corners
    filter_zerophase: bool = False  # the filter zerophase
    # * transforms, data agument
    train_trans: List[str] = field(default_factory=lambda: [
        "shift", "scale", "label"])  # the dataset transforms applied, can be shift, scale, and label
    val_trans: List[str] = field(default_factory=lambda: [
        "scale", "label"])  # the dataset transforms applied, can be shift, scale, and label
    test_trans: List[str] = field(default_factory=lambda: [
        "scale", "label"])  # the dataset transforms applied, can be shift, scale, and label

    # if we construct the stacking waveforms in training datset, only applied in training
    stack: bool = True
    stack_ratio: float = 0.9738217746076991  # hyper tune, the stack ratio
    min_stack_gap: int = 100  # the min gap between two stacked phases

    # if randomly replace the training waveform with the noise, only applied in training
    replace_noise: bool = True
    # hyper tune, the noise replacing ratio
    noise_replace_ratio: float = 0.3210340323794437

    scale_at_end: bool = True  # if we normalize the waveforms, applied on all the dataset
    # * batch size and shuffle
    train_batch_size: int = 32  # the training batch size
    val_batch_size: int = 1  # the validating batch size
    test_batch_size: int = 1  # the testing batch size
    train_shuffle: bool = True  # if we shuffle the dataset in training
    # * workers
    # dataloader workers, applied on all the dataloader (except the inference)
    num_workers: int = 5

    # * PS freq picked by Fan, used for visualizing
    load_ps_freq: bool = True  # if load the ps freq info in the h5 file


@dataclass
class SpectrogramConfig:
    """
    Set confiuration to generate the spectrogram from the waveform dataset
    """
    n_fft: int = 256  # hyper tune, use n_fft points to do fft in one sliding window
    # the distance between neighboring sliding window frames.
    hop_length: int = 1
    power: Optional[int] = None  # the power in doing stft, Non
    window_fn: str = "hann"  # the window to use when doing stft
    freqmin: float = 0.  # the output spec min freq
    freqmax: float = 10.  # the output spec max freq
    sampling_rate: int = 40  # the samping rate of the waveform
    height: int = 64  # the output spec height
    # should equal to win_len*sampling_rate (data.width), the output spec width
    width: int = 4800
    max_clamp: int = 3000  # hyper tune, the spec max clamp


@dataclass
class ModelConfig:
    """
    neural network model configuration
    """
    nn_model: str = "deeplabv3+"  # can be unet or deeplabv3+
    # the input channel to the model. Currently it's only affected by spec.power. When spec.power=2, change this to 3.
    in_channels: int = 6
    # the output channels, should be the number of phases plus 1 (noise output).
    out_channels: int = 4
    # n_freq is not used when train_with_spectrogram==False
    n_freq: int = 64  # should be the same as height in SpectrogramConfig
    # if False, train only use wave, should update other parameters by hand.
    train_with_spectrogram: bool = True

    # * below are unet configs, start with unet_
    unet_init_features: int = 32  # hyper tune, the feature number in Unet.
    # hyper tune, the repeating CNN layers for the first block
    unet_first_layer_repeating_cnn: int = 3
    unet_encoder_conv_kernel_size: List[int] = field(
        default_factory=lambda: [5, 5])  # hyper tune, the encoder kernel size
    unet_decoder_conv_kernel_size: List[int] = field(
        default_factory=lambda: [5, 5])  # hyper tune, the decoder kernel size
    # hyper tune, number of encoder/decoder blocks
    unet_encoder_decoder_depth: int = 5

    # * below are deeplab configs, start with deeplab_
    deeplab_encoder_name: Optional[str] = "resnet34"  # the backbone of deeplab
    deeplab_encoder_depth: Optional[int] = 5  # the encoder depth
    # the output stride of deeplab
    deeplab_encoder_output_stride: Optional[int] = 16
    # the decoder channels of deeplab
    deeplab_decoder_channels: Optional[int] = 256
    deeplab_decoder_atrous_rates: List[int] = field(default_factory=lambda: [
        12, 24, 36])  # the deeplab atrous rates
    # the final output upsamling ratio of deeplab
    deplab_upsampling: Optional[int] = 4


@dataclass
class TrainConfig:
    """
    the trainning configuration
    """
    # * random seed
    # if use deterministic algorithms. It can help with reproducibility. But it might fail on segmentation models.
    # see https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842
    deterministic: bool = False
    use_random_seed: bool = True  # if we want to use the random seed
    random_seed: int = 666  # the random seed
    # * basic configs
    learning_rate: float = 0.001  # hyper tune, the learning rate
    weight_decay: float = 1e-3  # hyper tune, the penalty term ratio for L2 norm
    epochs: int = 160  # the number of epochs (at max) to run
    sync_batchnorm: bool = True  # if sync batch between different GPUs
    # * acceleration
    # the accelerator to use, can be cpu or gpu at the moment.
    accelerator: str = "cpu"
    # the strategy to use, will be None when accelerator=='cpu'
    strategy: Optional[str] = "ddp_find_unused_parameters_false"
    use_amp: bool = False  # if use amp, might be overflow if use v100
    # if use a100 (along with amp). When use a100 without amp, can set this to be False.
    use_a100: bool = False
    distributed_devices: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3])  # the GPUs to use, will be None when accelerator=='cpu'
    # * test on local
    # the train batches in the test mode
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None  # the val batches in the test mode
    # the test batches in the test mode
    limit_test_batches: Optional[int] = None
    # * logging
    log_every_n_steps: int = 1  # how often to log with steps
    # * loss func
    loss_func: str = "kl_div"  # the loss func to use, currently support kl_div and focal
    # * run_type, whether train or hyper_tune
    # the run type, only influence the hyper_tune behavior to decide if load train/val or test dataset.
    run_type: str = "train"

    # * optimizer
    step_lr_milestones: List[int] = field(
        default_factory=lambda: [30, 60, 90, 120])  # at which steps the lr is decayed
    step_lr_gamma: float = 0.6  # the decay ratio


@dataclass
class VisualizeConfig:
    """
    the visualization configuration
    """
    example_num: int = 8  # the number of figs to save in logging
    log_train: bool = False  # if log the training step
    log_val: bool = True  # if log the val step
    log_test: bool = True  # if log the test step
    log_epoch: Optional[int] = 30  # logging epochs step
    # when plotting the sgram, what's the max clamp threshold
    sgram_threshold: Optional[int] = 500
    # in test_step, if logging to a seprate dir. Will be used along with log_test.
    log_test_seprate_folder: bool = False
    log_test_seprate_folder_path: str = MISSING  # the folder dir.
    # It will influence the filter range in plotting waveforms.
    # can be all (filter based on data config), P (P max range/fixed range), PS (from the max horizontal amp in spec)
    plot_waveform_based_on: str = "all"


@dataclass
class PostProcessConfig:
    """
    post process the model output, such as finding peak, cal metrics etc.
    """
    # metrics
    # within metrics_dt_threshold seconds, we count the phase is correctly detected.
    metrics_dt_threshold: float = 1.0

    # peaks
    sensitive_heights: Dict[str, float] = field(
        default_factory=lambda: {
            "TP": 0.5,
            "TS": 0.5,
            "TPS": 0.3
        })  # above the threshold, we regard the phases to be detected
    sensitive_distances: Dict[str, float] = field(
        default_factory=lambda: {
            "TP": 5.0,
            "TS": 5.0,
            "TPS": 5.0
        })  # within sensitive_distances seconds, we extract the peaks

    # in test_step, if we save the temporary result to disk
    save_test_step_to_disk: bool = False
    test_step_save_path: str = MISSING  # the test_step dumping directory


@dataclass
class WandbConfig:
    """
    config wandb
    """
    job_name: str = "test"
    project_name: str = "PhaseNet-TF"
    model_log_freq: int = 500


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
    # if the unit is m instead of nm. Might have no effect, put it here for numerical stability.
    unit_is_m: bool = True

    # * checkpoint loading reference
    # if use local checkpoint, we have to specify one, and will not consider wandb checkpoints
    use_local_checkpoint: bool = False
    local_checkpoint_path: Path = MISSING # local checkpoint path, used when use_local_checkpoint==True
    wandb_checkpoint_reference: str = MISSING  # the w&b checkpoint reference path

    # * infrence output
    inference_output_dir: Path = MISSING
    save_prediction_stream: bool = False
    save_waveform_stream: bool = False
    save_phase_arrivals: bool = True  # will be saved to phase_arrivals.csv


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
