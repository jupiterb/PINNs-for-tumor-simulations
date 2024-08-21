import torch as th
from torch.utils.data import DataLoader

from typing import Sequence

from phynn.data.interface import DataInterfaceFactory, DataKey
from phynn.data.export import DataExportManager
from phynn.data.set.seq import SequenceSamplesDataset
from phynn.physics import Simulation


class PhysicsResiduumsSamplesDataset(SequenceSamplesDataset):
    def __init__(self, factory: DataInterfaceFactory) -> None:
        super().__init__(factory)
        self._residuums = factory.get_flat_interface(DataKey.RESIDUUMS)

    def __getitem__(self, index: int) -> Sequence[th.Tensor]:
        return [*super().__getitem__(index), self._residuums.get(index)]


def create_phy_residuums(
    export: DataExportManager,
    data: SequenceSamplesDataset,
    simulation: Simulation,
    params: Sequence[float],
    batch_size=64,
) -> None:
    image_shape = data[0][0].shape
    params_batch = th.stack([th.Tensor(params) for _ in range(batch_size)])
    dl = DataLoader(data, batch_size, shuffle=False)

    with export.get() as e:
        residuums = e.create_export(DataKey.RESIDUUMS, image_shape)

        for u_input, _, duration in dl:
            with th.no_grad():
                u_output = simulation(u_input, params_batch, duration)

            residuums.export(u_output)
