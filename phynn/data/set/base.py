from torch.utils.data import Dataset

from typing import Protocol, runtime_checkable

from phynn.data.interface import DataInterfaceFactory


@runtime_checkable
class DatasetProtocol(Protocol):
    def __init__(self, factory: DataInterfaceFactory) -> None: ...


class FactoryBasedDataset(Dataset, DatasetProtocol): ...
