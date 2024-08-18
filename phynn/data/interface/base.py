import torch as th

from abc import ABC, abstractmethod
from enum import Enum


class DataKey(Enum, str):
    IMAGES = "images"
    TIMES = "times"
    PARAMS = "params"
    RESIDUUMS = "residuum"


class FlatDataInterface(ABC):
    @abstractmethod
    def get(self, i: int) -> th.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()


class SequenceDataInterface(ABC):
    @abstractmethod
    def get(self, i: int, j: int) -> th.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def series_number(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def series_length(self) -> int:
        raise NotImplementedError()


class DataInterfaceFactory(ABC):
    @abstractmethod
    def get_flat_interface(self, key: str) -> FlatDataInterface:
        raise NotImplementedError()

    @abstractmethod
    def get_sequence_interface(self, key: str) -> SequenceDataInterface:
        raise NotImplementedError()


class FlatDataTensorInterface(FlatDataInterface):
    def __init__(self, data: th.Tensor) -> None:
        self._data = data

    def get(self, i: int) -> th.Tensor:
        return self._data[i]

    @property
    def size(self) -> int:
        return len(self._data)


class SequenceDataTensorInterface(SequenceDataInterface):
    def __init__(self, data: th.Tensor) -> None:
        self._data = data

    def get(self, i: int, j: int) -> th.Tensor:
        return self._data[i, j]

    @property
    def series_number(self) -> int:
        return len(self._data)

    @property
    def series_length(self) -> int:
        return self._data.shape[1]
