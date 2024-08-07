import torch.nn as nn

from dataclasses import dataclass
from typing import Type

from phynn.nn.base import SequentialNetCreator, SequentialNetParams


@dataclass
class ConvBlockParams:
    kernel_size: int = 3
    activation: Type[nn.Module] = nn.LeakyReLU
    batch_norm: bool = True
    stride: int = 1
    same_padding: bool = True
    rescale: int = 1
    dropout: float = 0.0


@dataclass
class _ConvType:
    transpose: bool
    upsample: bool
    rescale_on_begin: bool


class Conv(SequentialNetCreator[int, ConvBlockParams]):
    def __init__(
        self,
        transpose: bool = False,
        upsample: bool = False,
        rescale_on_begin: bool = False,
    ) -> None:
        self._conv_type = _ConvType(transpose, upsample, rescale_on_begin)

    def create(self, params: SequentialNetParams[int, ConvBlockParams]) -> nn.Module:
        conv = nn.Sequential()
        in_channels = params.in_space

        for out_channels, block_params in params:
            conv += self._create_block(in_channels, out_channels, block_params)

        return conv

    def _create_block(
        self, in_channels: int, out_channels: int, block_params: ConvBlockParams
    ) -> nn.Sequential:
        d = block_params
        t = self._conv_type

        padding = (d.kernel_size - 1) // 2 if d.same_padding else 1
        conv_cls = nn.ConvTranspose2d if t.transpose else nn.Conv2d

        block = nn.Sequential()

        def rescale():
            if t.upsample:
                block.append(nn.Upsample(scale_factor=d.rescale, mode="bilinear"))
            else:
                block.append(nn.MaxPool2d(kernel_size=d.rescale, stride=d.rescale))

        if d.rescale > 1 and t.rescale_on_begin:
            rescale()

        if d.dropout > 0:
            block.append(nn.Dropout(d.dropout))

        conv = conv_cls(in_channels, out_channels, d.kernel_size, d.stride, padding)
        block.append(conv)

        if d.batch_norm:
            block.append(nn.BatchNorm2d(out_channels))

        block.append(d.activation())

        if d.rescale > 1 and not t.rescale_on_begin:
            rescale()

        return block
