import torch as th
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import Callable

from phynn.data.interface import DataInterfaceFactory, DataKey
from phynn.data.set.base import FactoryBasedDataset
from phynn.data.export import DataExportManager


class FlatImagesDataset(FactoryBasedDataset):
    def __init__(self, factory: DataInterfaceFactory) -> None:
        self._images = factory.get_flat_interface(DataKey.IMAGES)

    def __len__(self) -> int:
        return self._images.size

    def __getitem__(self, index: int) -> th.Tensor:
        return self._images.get(index)


class SequenceImagesDataset(FactoryBasedDataset):
    def __init__(self, factory: DataInterfaceFactory) -> None:
        self._images = factory.get_sequence_interface(DataKey.IMAGES)

    def __len__(self) -> int:
        return self._images.series_length * self._images.series_number

    def __getitem__(self, index: int) -> th.Tensor:
        i = index // self._images.series_length
        j = index % self._images.series_length
        return self._images.get(i, j)


ImagesDataset = FlatImagesDataset | SequenceImagesDataset


def preprocess_img(
    data: ImagesDataset,
    func: Callable[[th.Tensor], th.Tensor],
    export: DataExportManager,
    batch_size: int = 64,
) -> None:
    with export.get() as e:
        images = None
        dl = DataLoader(data, batch_size, shuffle=False)

        for batch in tqdm(dl, "Processing"):
            batch = func(batch)

            if images is None:
                image_shape = batch.shape[1:]
                images = e.create_export(DataKey.IMAGES, image_shape)

            images.export(batch)
