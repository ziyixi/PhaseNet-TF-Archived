"""
unet.py

An Unet modificaion for the spectrogram based seismic phase picking
"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from phasenet.conf.load_conf import Config
from phasenet.utils.spectrogram import spectrogram
from torch.nn.modules.loss import _WeightedLoss


class WeightedLoss(_WeightedLoss):
    def __init__(self, weight=None):
        super().__init__(weight=weight)
        self.weight = weight

    def forward(self, input, target):

        log_pred = nn.functional.log_softmax(input, 1)

        if self.weight is not None:
            target = target * \
                self.weight.unsqueeze(1).unsqueeze(1) / self.weight.sum()

        return -(target * log_pred).sum(dim=1).mean()


class UNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        cfg_model = cfg.model
        self._out_features = ["out"]
        self._out_feature_channels = {"out": cfg_model.out_channels}
        self._out_feature_strides = {"out": 1}

        self.use_stft = cfg_model.use_stft
        if self.use_stft:
            self.spectrogram = partial(
                spectrogram, cfg_spec=cfg.spectrogram)

        features = cfg_model.init_features
        in_channels = cfg_model.in_channels
        if self.use_stft:
            in_channels *= 2  # real amd imagenary parts
        self.encoder1 = UNet._block(
            in_channels, features, kernel_size=tuple(cfg_model.encoder_kernel_size), padding=tuple(cfg_model.encoder_padding), name="enc1"
        )
        #! note we add padding to pool
        self.pool1 = nn.MaxPool2d(
            kernel_size=tuple(cfg_model.encoder_kernel_size), stride=tuple(cfg_model.encoder_stride)
        )
        if self.use_stft:
            self.fc1 = nn.Sequential(
                nn.Linear(cfg_model.n_freq, 1), nn.ReLU(inplace=True))
        self.encoder2 = UNet._block(
            features, features * 2, kernel_size=tuple(cfg_model.encoder_kernel_size), padding=tuple(cfg_model.encoder_padding), name="enc2"
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=tuple(cfg_model.encoder_kernel_size), stride=tuple(cfg_model.encoder_stride))
        if self.use_stft:
            self.fc2 = nn.Sequential(
                nn.Linear(cfg_model.n_freq // 2, 1), nn.ReLU(inplace=True))
        self.encoder3 = UNet._block(
            features * 2, features * 4, kernel_size=tuple(cfg_model.encoder_kernel_size), padding=tuple(cfg_model.encoder_padding), name="enc3"
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=tuple(cfg_model.encoder_kernel_size), stride=tuple(cfg_model.encoder_stride))
        if self.use_stft:
            self.fc3 = nn.Sequential(
                nn.Linear(13, 1), nn.ReLU(inplace=True))
        self.encoder4 = UNet._block(
            features * 4, features * 8, kernel_size=tuple(cfg_model.encoder_kernel_size), padding=tuple(cfg_model.encoder_padding), name="enc4"
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=tuple(cfg_model.encoder_kernel_size), stride=tuple(cfg_model.encoder_stride))
        if self.use_stft:
            self.fc4 = nn.Sequential(
                nn.Linear(6, 1), nn.ReLU(inplace=True))

        self.bottleneck = UNet._block(
            features * 8, features * 16, kernel_size=tuple(cfg_model.encoder_kernel_size), padding=tuple(cfg_model.encoder_padding), name="bottleneck"
        )
        if self.use_stft:
            self.fc5 = nn.Sequential(
                nn.Linear(2, 1), nn.ReLU())

        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(features * 16, features * 8,
                               kernel_size=tuple(cfg_model.decoder_kernel_size), stride=tuple(cfg_model.decoder_stride)),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = UNet._block(
            (features * 8) * 2, features * 8, kernel_size=tuple(cfg_model.decoder_kernel_size), padding=tuple(cfg_model.decoder_padding), name="dec4"
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4,
                               kernel_size=tuple(cfg_model.decoder_kernel_size), stride=tuple(cfg_model.decoder_stride)),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = UNet._block(
            (features * 4) * 2, features * 4, kernel_size=tuple(cfg_model.decoder_kernel_size), padding=tuple(cfg_model.decoder_padding), name="dec3"
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2,
                               kernel_size=tuple(cfg_model.decoder_kernel_size), stride=tuple(cfg_model.decoder_stride)),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = UNet._block(
            (features * 2) * 2, features * 2, kernel_size=tuple(cfg_model.decoder_kernel_size), padding=tuple(cfg_model.decoder_padding), name="dec2"
        )
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, features, kernel_size=tuple(cfg_model.decoder_kernel_size), stride=tuple(cfg_model.decoder_stride)),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = UNet._block(
            features * 2, features, kernel_size=tuple(cfg_model.decoder_kernel_size), padding=tuple(cfg_model.decoder_padding), name="dec1"
        )

        self.conv = nn.Conv2d(in_channels=features,
                              out_channels=cfg_model.out_channels, kernel_size=1)

    def forward(self, x):

        #! here we remove the last dimension
        bt, ch, nt = x.shape

        if self.use_stft:
            sgram = x  # bt, ch, nt
            # bt*ch, nf, nframe, 2
            sgram = self.spectrogram(sgram.view(-1, nt))
            # bt, ch, nf, nframe, 2
            sgram = sgram.view(bt, ch, *sgram.shape[-3:])
            # 2, bt, ch, nf, nframe, 1
            components = sgram.split(1, dim=-1)
            # bt, 2*ch, nf, nframe
            sgram = torch.cat([components[1].squeeze(-1),
                               components[0].squeeze(-1)], dim=1)
            # bt, 2*ch, nframe, nf
            sgram = sgram.transpose(-1, -2)
            x = sgram

        print(f"{x.shape =}")
        enc1 = self.encoder1(x)
        print(f"{enc1.shape =}")
        enc2 = self.encoder2(self.pool1(enc1))
        print(f"{self.pool1(enc1).shape =}")
        print(f"{enc2.shape =}")
        if self.use_stft:
            enc1 = self.fc1(enc1)
        enc3 = self.encoder3(self.pool2(enc2))
        print(f"{self.pool2(enc2).shape =}")
        print(f"{enc3.shape =}")
        if self.use_stft:
            enc2 = self.fc2(enc2)
        enc4 = self.encoder4(self.pool3(enc3))
        if self.use_stft:
            enc3 = self.fc3(enc3)

        bottleneck = self.bottleneck(self.pool4(enc4))
        if self.use_stft:
            enc4 = self.fc4(enc4)
            bottleneck = self.fc5(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = UNet._cat(dec4, enc4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = UNet._cat(dec3, enc3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = UNet._cat(dec2, enc2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = UNet._cat(dec1, enc1)
        dec1 = self.decoder1(dec1)
        # return torch.sigmoid(self.conv(dec1))
        out = self.conv(dec1)
        print(f"{out.shape =}")
        result = {}
        result["out"] = F.interpolate(out, size=(
            nt, out.shape[-1]), mode='bilinear', align_corners=False)
        if self.use_stft:
            result["sgram"] = sgram
        return result

    @ staticmethod
    def _cat(x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                        2, diffY // 2, diffY - diffY // 2])

        x1 = torch.cat((x1, x2), dim=1)
        return x1

    @ staticmethod
    def _block(in_channels, features, kernel_size=(3, 3), padding=(1, 1), name=""):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
