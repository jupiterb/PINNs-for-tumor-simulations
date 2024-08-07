from __future__ import annotations

import torch.nn as nn

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterator, Type, TypeVar


SpaceParams = TypeVar("SpaceParams")
BlockParams = TypeVar("BlockParams")


@dataclass
class SequentialNetParams(Generic[SpaceParams, BlockParams]):
    _spaces: list[SpaceParams]
    _blocks: list[BlockParams]

    @property
    def in_space(self) -> SpaceParams:
        return self._spaces[0]

    def __iter__(self) -> Iterator[tuple[SpaceParams, BlockParams]]:
        return zip(self._spaces[1:], self._blocks)

    def __add__(
        self, other: SequentialNetParams[SpaceParams, BlockParams]
    ) -> SequentialNetParams[SpaceParams, BlockParams]:
        spaces = self._spaces + other._spaces
        blocks = self._blocks + other._blocks
        return SequentialNetParams(spaces, blocks)


class SequentialNetParamsFactory(Generic[SpaceParams, BlockParams]):
    @staticmethod
    def init(space: SpaceParams) -> SequentialNetParams[SpaceParams, BlockParams]:
        return SequentialNetParams([space], [])

    @staticmethod
    def layer(
        space: SpaceParams, block: BlockParams
    ) -> SequentialNetParams[SpaceParams, BlockParams]:
        return SequentialNetParams([space], [block])


class SequentialNetCreator(Generic[SpaceParams, BlockParams], ABC):
    @abstractmethod
    def create(
        self, params: SequentialNetParams[SpaceParams, BlockParams]
    ) -> nn.Module:
        raise NotImplementedError()


def get_factory(
    creator: SequentialNetCreator[SpaceParams, BlockParams]
) -> Type[SequentialNetParamsFactory[SpaceParams, BlockParams]]:
    return SequentialNetParamsFactory[SpaceParams, BlockParams]
