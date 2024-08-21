import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Sequence, Type

from phynn.nn.conv import (
    ConvNet,
    ConvNetParams,
    ConvNetBlockParams,
    ConvNetCommonParams,
)


@dataclass
class UNetParams:
    in_channels: int
    out_channels: int
    out_activation: Type[nn.Module]
    levels_channels: Sequence[int]


class UNetLevel(nn.Module):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, sublevel: nn.Module
    ) -> None:
        super(UNetLevel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._sublevel = sublevel

    def forward(self, x: th.Tensor) -> th.Tensor:
        x_enc = self._encoder(x)
        x_sub = self._sublevel(x_enc)
        x_cat = th.concat([x_enc, x_sub], dim=1)
        return self._decoder(x_cat)


class UNet(nn.Module):
    def __init__(self, params: UNetParams) -> None:
        if len(params.levels_channels) < 2:
            raise ValueError(
                "UNet should have at least two levels, so len(levels_channels) should be >= 2."
            )

        super(UNet, self).__init__()

        unet_in_params = ConvNetParams(
            [params.in_channels, params.levels_channels[0], params.levels_channels[0]],
            [ConvNetBlockParams()],
            ConvNetCommonParams(),
        )
        unet_in_conv = ConvNet(unet_in_params)

        unet_out_params = ConvNetParams(
            [
                params.levels_channels[0] + params.levels_channels[0],
                params.levels_channels[0],
                params.levels_channels[0],
                params.out_channels,
            ],
            [
                ConvNetBlockParams(),
                ConvNetBlockParams(),
                ConvNetBlockParams(kernel_size=1, activation=params.out_activation),
            ],
            ConvNetCommonParams(),
        )
        unet_out_conv = ConvNet(unet_out_params)

        sublevel = UNet._build_level(params.levels_channels)

        self._unet = UNetLevel(unet_in_conv, unet_out_conv, sublevel)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._unet(x)

    @staticmethod
    def _build_level(levels_channels: Sequence[int]) -> nn.Module:
        if len(levels_channels) > 2:
            return UNet._build_unet_level(levels_channels)
        else:
            return UNet._build_bridge_level(*levels_channels)

    @staticmethod
    def _build_unet_level(levels_channels: Sequence[int]) -> nn.Module:
        in_channels, level_channels = levels_channels[0], levels_channels[1]
        concat_channels = 2 * level_channels

        encoder_params = ConvNetParams(
            [in_channels, level_channels, level_channels],
            [ConvNetBlockParams(rescale=2), ConvNetBlockParams()],
            ConvNetCommonParams(rescale_on_begin=True),
        )
        encoder = ConvNet(encoder_params)

        decoder_params = ConvNetParams(
            [concat_channels, level_channels, in_channels],
            [ConvNetBlockParams(), ConvNetBlockParams(rescale=2)],
            ConvNetCommonParams(upsample=True),
        )
        decoder = ConvNet(decoder_params)

        sublevel = UNet._build_level(levels_channels[1:])

        return UNetLevel(encoder, decoder, sublevel)

    @staticmethod
    def _build_bridge_level(in_channels: int, level_channels: int) -> nn.Module:
        encoder_params = ConvNetParams(
            [in_channels, level_channels],
            [ConvNetBlockParams(rescale=2)],
            ConvNetCommonParams(rescale_on_begin=True),
        )
        encoder = ConvNet(encoder_params)

        decoder_params = ConvNetParams(
            [level_channels, in_channels],
            [ConvNetBlockParams(rescale=2)],
            ConvNetCommonParams(upsample=True),
        )
        decoder = ConvNet(decoder_params)

        return nn.Sequential(encoder, decoder)
