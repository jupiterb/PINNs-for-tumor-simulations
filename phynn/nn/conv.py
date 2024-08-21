import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Sequence, Type


@dataclass
class ConvNetCommonParams:
    transpose: bool = False
    upsample: bool = False
    rescale_on_begin: bool = False


@dataclass
class ConvNetBlockParams:
    kernel_size: int = 3
    activation: Type[nn.Module] = nn.LeakyReLU
    batch_norm: bool = True
    stride: int = 1
    same_padding: bool = True
    rescale: int = 1
    dropout: float = 0.0


@dataclass
class ConvNetParams:
    channels: Sequence[int]
    blocks: Sequence[ConvNetBlockParams]
    common: ConvNetCommonParams


def ConvBlock(
    in_channels: int,
    out_channels: int,
    block_params: ConvNetBlockParams,
    common_params: ConvNetCommonParams,
) -> nn.Sequential:
    b = block_params
    c = common_params

    padding = (b.kernel_size - 1) // 2 if b.same_padding else 1
    conv_cls = nn.ConvTranspose2d if c.transpose else nn.Conv2d

    block = nn.Sequential()

    def rescale():
        if c.upsample:
            block.append(nn.Upsample(scale_factor=b.rescale, mode="bilinear"))
        else:
            block.append(nn.MaxPool2d(kernel_size=b.rescale, stride=b.rescale))

    if b.rescale > 1 and c.rescale_on_begin:
        rescale()

    if b.dropout > 0:
        block.append(nn.Dropout(b.dropout))

    conv = conv_cls(in_channels, out_channels, b.kernel_size, b.stride, padding)
    block.append(conv)

    if b.batch_norm:
        block.append(nn.BatchNorm2d(out_channels))

    block.append(b.activation())

    if b.rescale > 1 and not c.rescale_on_begin:
        rescale()

    return block


class ConvNet(nn.Module):
    def __init__(self, params: ConvNetParams) -> None:
        super(ConvNet, self).__init__()
        self._conv = nn.Sequential()
        in_channels = params.channels[0]

        for out_channels, block_params in zip(params.channels[1:], params.blocks):
            block = ConvBlock(in_channels, out_channels, block_params, params.common)
            self._conv += block
            in_channels = out_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._conv(x)
