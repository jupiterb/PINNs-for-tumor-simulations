import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Generic

from phynn.nn.base import (
    SequentialNetCreator,
    SequentialNetParams,
    SpaceParams,
    BlockParams,
    get_factory,
)


class ResBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super(ResBlock, self).__init__()
        self._block = block

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._block(x) + x


@dataclass
class ResBlockParams(Generic[BlockParams]):
    params: BlockParams
    size: int


MaybeResBlockParams = ResBlockParams[BlockParams] | BlockParams


class ResNet(SequentialNetCreator[SpaceParams, MaybeResBlockParams[BlockParams]]):
    def __init__(
        self, net_creator: SequentialNetCreator[SpaceParams, BlockParams]
    ) -> None:
        self._net_creator = net_creator

    def create(
        self,
        params: SequentialNetParams[SpaceParams, MaybeResBlockParams[BlockParams]],
    ) -> nn.Module:
        resnet = nn.Sequential()
        in_space = params.in_space

        for out_space, block_params in params:
            block = self._create_block(in_space, out_space, block_params)
            resnet.append(block)

        return resnet

    def _create_block(
        self,
        in_space: SpaceParams,
        out_space: SpaceParams,
        block_params: MaybeResBlockParams[BlockParams],
    ) -> nn.Module:
        factory = get_factory(self._net_creator)
        params = factory.init(in_space)

        if isinstance(block_params, ResBlockParams):
            for _ in range(block_params.size):
                params += factory.layer(out_space, block_params.params)
            return ResBlock(self._net_creator.create(params))
        else:
            params += factory.layer(out_space, block_params)
            return self._net_creator.create(params)
