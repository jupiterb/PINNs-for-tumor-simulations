import torch as th
from torch.utils.data import DataLoader

from typing import Callable, Sequence

from phynn.data.interface import DataInterfaceFactory, DataKey
from phynn.data.set.base import FactoryBasedDataset
from phynn.data.export import DataExportManager


class SequenceDataset(FactoryBasedDataset):
    def __init__(self, factory: DataInterfaceFactory) -> None:
        self._images = factory.get_flat_interface(DataKey.IMAGES)
        self._times = factory.get_flat_interface(DataKey.TIMES)

    def __len__(self) -> int:
        return self._images.size

    def __getitem__(self, index: int) -> tuple[th.Tensor, th.Tensor]:
        return self._images.get(index), self._times.get(index)


class SequenceSamplesDataset(FactoryBasedDataset):
    def __init__(self, factory: DataInterfaceFactory) -> None:
        self._images = factory.get_sequence_interface(DataKey.IMAGES)
        self._times = factory.get_sequence_interface(DataKey.TIMES)

        self._series_number = self._images.series_number
        self._series_length = self._images.series_length

        self._series_samples = (
            (self._times.series_length - 1) * self._series_length // 2
        )

        self._start_ixs = []
        self._result_ixs = []

        for i in range(self._series_length - 1):
            for j in range(i, self._series_length):
                self._start_ixs.append(i)
                self._result_ixs.append(j)

    def __len__(self) -> int:
        return self._series_number * self._series_samples

    def __getitem__(self, index: int) -> Sequence[th.Tensor]:
        return self._get_from_sequences(*self._decompose_index(index))

    def _decompose_index(self, index: int) -> tuple[int, int, int]:
        series_index = index // self._series_samples
        sample_ixs = index % self._series_samples
        return series_index, self._start_ixs[sample_ixs], self._result_ixs[sample_ixs]

    def _get_from_sequences(
        self, series_index: int, start_index: int, end_index: int
    ) -> Sequence[th.Tensor]:
        return (
            self._images.get(series_index, start_index),
            self._images.get(series_index, end_index),
            self._times.get(series_index, end_index)
            - self._times.get(series_index, start_index),
        )


def preprocess_seq(
    data: SequenceDataset,
    func: Callable[[th.Tensor], th.Tensor],
    export: DataExportManager,
    batch_size: int = 64,
) -> None:
    image_shape = data[0][0].shape[1:]
    time_shape = data[0][1].shape[1:]

    dl = DataLoader(data, batch_size, shuffle=False)

    with export.get() as e:
        images = e.create_export(DataKey.IMAGES, image_shape)
        times = e.create_export(DataKey.TIMES, time_shape)

        for images_batch, times_batch in dl:
            images_batch = func(images_batch)
            images.export(images_batch)
            times.export(times_batch)
