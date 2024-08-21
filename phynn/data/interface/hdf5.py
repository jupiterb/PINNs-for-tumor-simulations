import h5py
import os

import torch as th

from phynn.data.interface.base import (
    DataInterfaceFactory,
    FlatDataInterface,
    FlatDataTensorInterface,
    SequenceDataInterface,
    SequenceDataTensorInterface,
)


class HDF5DataInterfaceFactory(DataInterfaceFactory):
    def __init__(
        self,
        path: os.PathLike,
        device: th.device,
        data_type: th.dtype = th.float32,
    ) -> None:
        self._path = path
        self._device = device
        self._data_type = data_type

    def get_flat_interface(self, key: str) -> FlatDataInterface:
        data = self._get_data(key)
        return FlatDataTensorInterface(data)

    def get_sequence_interface(self, key: str) -> SequenceDataInterface:
        data = self._get_data(key)
        return SequenceDataTensorInterface(data)

    def _get_data(self, key: str) -> th.Tensor:
        with h5py.File(self._path, "r") as file:
            data = th.tensor(file[key][:], dtype=self._data_type).to(self._device)  # type: ignore
        return data
