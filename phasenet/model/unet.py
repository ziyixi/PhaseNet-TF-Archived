"""
unet.py

An Unet modificaion for the spectrogram based seismic phase picking
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from phasenet.conf.load_conf import ModelConfig


class UNet(nn.Module):
    def __init__(self, cfg_model: ModelConfig):
        super().__init__()
        # * modle feature
        self._out_features = ["out"]
        self._out_feature_channels = {"out": cfg_model.out_channels}
        self._out_feature_strides = {"out": 1}
        self.more_layer = cfg_model.more_layer

        # * enc1
        features = cfg_model.init_features
        in_channels = cfg_model.in_channels
        if cfg_model.first_layer_more_cnn:
            self.encoder1 = UNet._block_firstlayer(
                in_channels, features, kernel_size=tuple(cfg_model.encoder_conv_kernel_size),  name="enc1"
            )
        else:
            self.encoder1 = UNet._block(
                in_channels, features, kernel_size=tuple(cfg_model.encoder_conv_kernel_size),  name="enc1"
            )
        self.pool1 = nn.MaxPool2d(
            kernel_size=tuple(cfg_model.encoder_pool_kernel_size), stride=tuple(cfg_model.encoder_pool_stride)
        )

        # * enc2
        self.encoder2 = UNet._block(
            features, features * 2, kernel_size=tuple(cfg_model.encoder_conv_kernel_size),  name="enc2"
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=tuple(cfg_model.encoder_pool_kernel_size), stride=tuple(cfg_model.encoder_pool_stride))

        # * enc3
        self.encoder3 = UNet._block(
            features * 2, features * 4, kernel_size=tuple(cfg_model.encoder_conv_kernel_size),  name="enc3"
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=tuple(cfg_model.encoder_pool_kernel_size), stride=tuple(cfg_model.encoder_pool_stride))

        # * enc4
        self.encoder4 = UNet._block(
            features * 4, features * 8, kernel_size=tuple(cfg_model.encoder_conv_kernel_size),  name="enc4"
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=tuple(cfg_model.encoder_pool_kernel_size), stride=tuple(cfg_model.encoder_pool_stride))

        # * possible enc5
        if cfg_model.more_layer:
            self.encoder5 = UNet._block(
                features * 8, features * 16, kernel_size=tuple(cfg_model.encoder_conv_kernel_size),  name="enc5"
            )
            self.pool5 = nn.MaxPool2d(
                kernel_size=tuple(cfg_model.encoder_pool_kernel_size), stride=tuple(cfg_model.encoder_pool_stride))

        # * bottleneck (possible enc5 incluence)
        if cfg_model.more_layer:
            self.bottleneck = UNet._block(
                features * 16, features * 32, kernel_size=tuple(cfg_model.encoder_conv_kernel_size),  name="bottleneck"
            )
        else:
            self.bottleneck = UNet._block(
                features * 8, features * 16, kernel_size=tuple(cfg_model.encoder_conv_kernel_size),  name="bottleneck"
            )

        # * possible dec5
        if cfg_model.more_layer:
            self.upconv5 = nn.Sequential(
                nn.ConvTranspose2d(features * 32, features * 16,
                                   kernel_size=tuple(cfg_model.decoder_pool_kernel_size), stride=tuple(cfg_model.decoder_pool_stride)),
                nn.ReLU(inplace=True),
            )
            self.decoder5 = UNet._block(
                (features * 16) * 2, features * 16, kernel_size=tuple(cfg_model.decoder_conv_kernel_size), name="dec5"
            )

        # * dec4
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(features * 16, features * 8,
                               kernel_size=tuple(cfg_model.decoder_pool_kernel_size), stride=tuple(cfg_model.decoder_pool_stride)),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = UNet._block(
            (features * 8) * 2, features * 8, kernel_size=tuple(cfg_model.decoder_conv_kernel_size), name="dec4"
        )

        # * dec3
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4,
                               kernel_size=tuple(cfg_model.decoder_pool_kernel_size), stride=tuple(cfg_model.decoder_pool_stride)),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = UNet._block(
            (features * 4) * 2, features * 4, kernel_size=tuple(cfg_model.decoder_conv_kernel_size),  name="dec3"
        )

        # * dec2
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2,
                               kernel_size=tuple(cfg_model.decoder_pool_kernel_size), stride=tuple(cfg_model.decoder_pool_stride)),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = UNet._block(
            (features * 2) * 2, features * 2, kernel_size=tuple(cfg_model.decoder_conv_kernel_size), name="dec2"
        )

        # * dec1
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, features, kernel_size=tuple(cfg_model.decoder_pool_kernel_size), stride=tuple(cfg_model.decoder_pool_stride)),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = UNet._block(
            features * 2, features, kernel_size=tuple(cfg_model.decoder_conv_kernel_size),  name="dec1"
        )

        # * output
        self.conv = nn.Conv2d(in_channels=features,
                              out_channels=cfg_model.out_channels, kernel_size=(1, cfg_model.n_freq))

    def forward(self, x: torch.Tensor):
        # the input is a batch of spectrograms, expected nf=16k
        bt, ch, nf, nt = x.shape
        # the sgram need to be rotated to nt,nf we can map frequency to size 1
        x = x.transpose(-1, -2)

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))

        if self.more_layer:
            enc5 = self.encoder5(self.pool4(enc4))

        if self.more_layer:
            bottleneck = self.bottleneck(self.pool5(enc5))
        else:
            bottleneck = self.bottleneck(self.pool4(enc4))

        if self.more_layer:
            dec5_uc = self.upconv5(bottleneck)
            dec5_ct = UNet._cat(dec5_uc, enc5)
            dec5 = self.decoder5(dec5_ct)

        if self.more_layer:
            dec4_uc = self.upconv4(dec5)
        else:
            dec4_uc = self.upconv4(bottleneck)
        dec4_ct = UNet._cat(dec4_uc, enc4)
        dec4 = self.decoder4(dec4_ct)

        dec3_uc = self.upconv3(dec4)
        dec3_ct = UNet._cat(dec3_uc, enc3)
        dec3 = self.decoder3(dec3_ct)

        dec2_uc = self.upconv2(dec3)
        dec2_ct = UNet._cat(dec2_uc, enc2)
        dec2 = self.decoder2(dec2_ct)

        dec1_uc = self.upconv1(dec2)
        dec1_ct = UNet._cat(dec1_uc, enc1)
        dec1 = self.decoder1(dec1_ct)

        out = self.conv(dec1)
        # change out from nt,1 to nt
        # so here the out will be bt, ch, nt
        # we design nt to be npts, so no need to upsampling
        out = out.squeeze(-1)

        # prepare result
        result = {}
        # due to kernel size, the out's nt might not be the exact nt
        # result["predict"] = F.interpolate(
        #     out, size=nt, mode='linear', align_corners=False)
        result["predict"] = out
        return result

    @staticmethod
    def _cat(x1, x2):
        # assume the size of x1 is smaller than x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # padding is from the last dimension
        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                        2, diffY // 2, diffY - diffY // 2])

        x1 = torch.cat((x1, x2), dim=1)
        return x1

    @staticmethod
    def _block(in_channels, features, kernel_size=(3, 3), name=""):
        return nn.Sequential(OrderedDict([
            (
                name + "_conv1",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding='same',
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
                    padding='same',
                    bias=False,
                ),
            ),
            (name + "_norm2", nn.BatchNorm2d(num_features=features)),
            (name + "_relu2", nn.ReLU(inplace=True)),
        ]))

    @staticmethod
    def _block_firstlayer(in_channels, features, kernel_size=(3, 3), name=""):
        return nn.Sequential(OrderedDict([
            (
                name + "_conv1",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding='same',
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
                    padding='same',
                    bias=False,
                ),
            ),
            (name + "_norm2", nn.BatchNorm2d(num_features=features)),
            (name + "_relu2", nn.ReLU(inplace=True)),

            (
                name + "_conv3",
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding='same',
                    bias=False,
                ),
            ),
            (name + "_norm3", nn.BatchNorm2d(num_features=features)),
            (name + "_relu3", nn.ReLU(inplace=True)),

            (
                name + "_conv4",
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding='same',
                    bias=False,
                ),
            ),
            (name + "_norm4", nn.BatchNorm2d(num_features=features)),
            (name + "_relu4", nn.ReLU(inplace=True)),
        ]))
