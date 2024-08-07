from __future__ import annotations

import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Generic, Iterator

from phynn.nn.base import (
    SequentialNetCreator,
    SequentialNetParams,
    SpaceParams,
    BlockParams,
    get_factory,
)


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


@dataclass
class UNetLevelParams(Generic[SpaceParams, BlockParams]):
    concat_space_params: SpaceParams
    block_params: BlockParams


MaybeUNetLevelParams = UNetLevelParams[SpaceParams, BlockParams] | BlockParams


class UNet(
    SequentialNetCreator[SpaceParams, MaybeUNetLevelParams[SpaceParams, BlockParams]]
):
    def __init__(
        self,
        encoder_creator: SequentialNetCreator[SpaceParams, BlockParams],
        decoder_creator: SequentialNetCreator[SpaceParams, BlockParams],
    ) -> None:
        self._encoder_creator = encoder_creator
        self._decoder_creator = decoder_creator

    def create(
        self,
        params: SequentialNetParams[
            SpaceParams, MaybeUNetLevelParams[SpaceParams, BlockParams]
        ],
    ) -> nn.Module:
        return self._create_level(params.in_space, params.__iter__())

    def _create_level(
        self,
        in_space: SpaceParams,
        params_iter: Iterator[
            tuple[SpaceParams, MaybeUNetLevelParams[SpaceParams, BlockParams]]
        ],
    ) -> nn.Module:
        factory = get_factory(self._encoder_creator)
        encoder_params = factory.init(in_space)
        decoder_params = factory.init(in_space)
        sublevel = None

        while True:
            try:
                out_space, unet_block_params = next(params_iter)

                match unet_block_params:
                    case UNetLevelParams(concat_space_params, block_params):
                        encoder_params += factory.layer(out_space, block_params)
                        decoder_params = (
                            factory.layer(concat_space_params, block_params)
                            + decoder_params
                        )
                        sublevel = self._create_level(concat_space_params, params_iter)
                    case block_params:
                        encoder_params += factory.layer(out_space, block_params)
                        decoder_params = (
                            factory.layer(out_space, block_params) + decoder_params
                        )

                in_space = out_space

            except StopIteration:
                break

        encoder = self._encoder_creator.create(encoder_params)
        decoder = self._decoder_creator.create(decoder_params)

        if sublevel is None:
            return nn.Sequential(encoder, decoder)
        else:
            return UNetLevel(encoder, decoder, sublevel)
