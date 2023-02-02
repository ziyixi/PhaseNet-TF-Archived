"""
segmentation_models.py

Get models from https://github.com/qubvel/segmentation_models.pytorch
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from phasenet.conf import ModelConfig


def create_smp_model(model_conf: ModelConfig) -> nn.Module:
    if model_conf.nn_model == "deeplabv3+":
        image_model = smp.DeepLabV3Plus(
            encoder_name=model_conf.deeplab_encoder_name,
            encoder_depth=model_conf.deeplab_encoder_depth,
            encoder_weights=None,
            encoder_output_stride=model_conf.deeplab_encoder_output_stride,
            decoder_channels=model_conf.deeplab_decoder_channels,
            decoder_atrous_rates=tuple(
                model_conf.deeplab_decoder_atrous_rates),
            in_channels=model_conf.in_channels,
            classes=model_conf.out_channels,
            upsampling=model_conf.deplab_upsampling
        )
    else:
        raise Exception(
            f"the model {model_conf.deeplab_model_name} is not implemented!")

    # * convert the 2d output image to 1d time series data
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # self.bn = nn.BatchNorm2d(model_conf.in_channels)
            self.image_model = image_model
            self.fc1 = nn.Linear(model_conf.n_freq, model_conf.n_freq//2)
            self.fc2 = nn.Linear(model_conf.n_freq//2, 1)

        def forward(self, x: torch.Tensor):
            result = {}
            # x = self.bn(x)
            x = F.relu(self.image_model(x))
            result["segout"] = x.clone()
            # transpose to have the last axis as freq, as Linear required
            x = x.transpose(-1, -2)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = x.squeeze(-1)

            # * prepare result
            result["predict"] = x
            return result

    return Model
