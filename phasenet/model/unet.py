"""
unet.py

An Unet modificaion for the spectrogram based seismic phase picking
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# * ============== define blocks ============== * #
class RepeatingConv(nn.Module):
    """CNN(i->f) => BN => ReLU => (CNN(f->f) => BN => ReLU)*r"""

    def __init__(self, i: int, f: int, r: int, ksize: Tuple[int, int]) -> None:
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=i, out_channels=f, kernel_size=ksize, padding='same', bias=False,
            ),
            nn.BatchNorm2d(num_features=f),
            nn.ReLU(inplace=True),
        )
        # repeating convs
        repeating_convs = []
        for _ in range(r):
            repeating_convs.extend([
                nn.Conv2d(
                    in_channels=f, out_channels=f, kernel_size=ksize, padding='same', bias=False,
                ),
                nn.BatchNorm2d(num_features=f),
                nn.ReLU(inplace=True),
            ])
        self.second_conv = nn.Sequential(*repeating_convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_conv(self.first_conv(x))


class Down(nn.Module):
    """RepeatingConv => (MaxPool2d , Sequential(h->1))"""

    def __init__(self, i: int, f: int, r: int, h: int, ksize: Tuple[int, int]) -> None:
        super().__init__()
        # repeating conv
        self.repeating_conv = RepeatingConv(i, f, r, ksize)
        # fc and maxpooling
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc = nn.Sequential(
            nn.Linear(h, 1),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.repeating_conv(x)
        return self.pool(x), self.fc(x)


class BottleNeck(nn.Module):
    """RepeatingConv => Sequential(h->1)"""

    def __init__(self, i: int, f: int, r: int, h: int, ksize: Tuple[int, int]) -> None:
        super().__init__()
        # repeating conv
        self.repeating_conv = RepeatingConv(i, f, r, ksize)
        # fc
        self.fc = nn.Sequential(
            nn.Linear(h, 1),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.repeating_conv(x)
        return self.fc(x)


class Up(nn.Module):
    def __init__(self, i: int, f: int, r: int, ksize: Tuple[int, int]) -> None:
        """(ConvTranspose2d => ReLU + skip) => RepeatingConv"""
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(i, f, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.decoder = RepeatingConv(2*f, f, r, ksize)

    def forward(self, x: torch.tensor, skip: torch.tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = self._cat(x, skip)
        return self.decoder(x)

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


# * ============== define U-Net model ============== * #

class UNet(nn.Module):
    def __init__(self, features: int, in_cha: int, out_cha: int, first_layer_repeating_cnn: int, n_freq: int, ksize_down: Tuple[int, int], ksize_up: Tuple[int, int]):
        super().__init__()
        # * encoders
        self.enc1 = Down(in_cha, features,
                         first_layer_repeating_cnn, n_freq, ksize_down)
        self.enc2 = Down(features*1, features*2, 1, n_freq//2, ksize_down)
        self.enc3 = Down(features*2, features*4, 1, n_freq//4, ksize_down)
        self.enc4 = Down(features*4, features*8, 1, n_freq//8, ksize_down)
        self.enc5 = Down(features*8, features*16, 1, n_freq//16, ksize_down)

        # * bottleneck
        self.bottleneck = BottleNeck(
            features*16, features*32, 1, n_freq//32, ksize_down)

        # * decoders
        self.dec5 = Up(features*32, features*16, 1, ksize_up)
        self.dec4 = Up(features*16, features*8, 1, ksize_up)
        self.dec3 = Up(features*8, features*4, 1, ksize_up)
        self.dec2 = Up(features*4, features*2, 1, ksize_up)
        self.dec1 = Up(features*2, features*1, 1, ksize_up)

        # * out
        self.conv_out = nn.Conv2d(
            in_channels=features, out_channels=out_cha, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the input is a batch of spectrograms, expected nf=32k, where k is positive
        # bt, ch, nf, nt = x.shape
        # the sgram need to be rotated to nt,nf we can map frequency to size 1
        x = x.transpose(-1, -2)

        # * encode
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        x, skip5 = self.enc5(x)

        # * bottleneck
        x = self.bottleneck(x)

        # * decode
        x = self.dec5(x, skip5)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # * output
        x = self.conv_out(x)
        # change out from nt,1 to nt
        # so here the out will be bt, ch, nt
        # we design nt to be npts, so no need to upsampling
        x = x.squeeze(-1)

        # * prepare result
        result = {}
        result["predict"] = x
        return result
