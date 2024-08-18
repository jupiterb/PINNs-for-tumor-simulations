import torch as th

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Sequence


class DataExporter(ABC):
    @abstractmethod
    def export(self, batch: th.Tensor) -> None:
        raise NotImplementedError()


class DataExporterFactory(ABC):
    @abstractmethod
    def create_export(self, name: str, data_shape: Sequence[int]) -> DataExporter:
        raise NotImplementedError()


class DataExportManager(ABC):
    @abstractmethod
    @contextmanager
    def get(self) -> Generator[DataExporterFactory, None, None]:
        raise NotImplementedError()
