from __future__ import annotations

import h5py
import os

import torch as th

from contextlib import contextmanager
from typing import Generator, Sequence


from phynn.data.export.base import (
    DataExporter,
    DataExporterFactory,
    DataExportManager,
)


class HDF5DataExporter(DataExporter):
    def __init__(self, dataset: h5py.Dataset) -> None:
        self._dataset = dataset

    def export(self, batch: th.Tensor) -> None:
        batch_size = batch.shape[0]
        current_size = self._dataset.shape[0]
        new_size = current_size + batch_size
        self._dataset.resize(new_size, axis=0)
        self._dataset[current_size:new_size, ...] = batch.detach().numpy()


class HDF5DataExporterFactory(DataExporterFactory):
    def __init__(self, file: h5py.File) -> None:
        self._file = file

    def create_export(self, name: str, data_shape: Sequence[int]) -> DataExporter:
        dataset = self._file.create_dataset(
            name,
            shape=(0, *data_shape),
            maxshape=(None, *data_shape),
            dtype="float32",
        )
        return HDF5DataExporter(dataset)


class HDF5DataExportManager(DataExportManager):
    def __init__(self, path: os.PathLike) -> None:
        self._path = path

    @contextmanager
    def get(self) -> Generator[DataExporterFactory, None, None]:
        dir_path = os.path.dirname(self._path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file = h5py.File(self._path, "a")

        yield HDF5DataExporterFactory(file)

        file.close()
