import torch as th
from torch.utils.data import Dataset, DataLoader

from typing import Callable

from phynn.data.interface import DataInterfaceFactory, DataKey
from phynn.data.export import DataExportManager


class ImagesDataset(Dataset):
    def __init__(self, factory: DataInterfaceFactory) -> None:
        super().__init__()
        self._images = factory.get_flat_interface(DataKey.IMAGES)

    def __len__(self) -> int:
        return self._images.size

    def __getitem__(self, index: int) -> th.Tensor:
        return self._images.get(index)


def preprocess_img(
    data: ImagesDataset,
    func: Callable[[th.Tensor], th.Tensor],
    export: DataExportManager,
    batch_size: int = 64,
) -> None:
    image_shape = data[0].shape
    dl = DataLoader(data, batch_size, shuffle=False)

    with export.get() as e:
        images = e.create_export(DataKey.IMAGES, image_shape)

        for batch in dl:
            batch = func(batch)
            images.export(batch)
