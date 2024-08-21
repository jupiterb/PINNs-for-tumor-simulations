import torch as th
from torch.utils.data import DataLoader

from itertools import product
from tqdm import tqdm
from typing import Callable, Sequence

from phynn.data.interface import DataInterfaceFactory, DataKey
from phynn.data.export import DataExportManager
from phynn.data.set.img import ImagesDataset
from phynn.data.set.seq import SequenceSamplesDataset
from phynn.physics import Simulation


class SimulationSamplesDataset(SequenceSamplesDataset):
    def __init__(self, factory: DataInterfaceFactory) -> None:
        super().__init__(factory)
        self._params = factory.get_flat_interface(DataKey.PARAMS)

    def _get_from_sequences(
        self, series_index: int, start_index: int, end_index: int
    ) -> Sequence[th.Tensor]:
        return [
            *super()._get_from_sequences(series_index, start_index, end_index),
            self._params.get(series_index),
        ]


def create_simulation(
    export: DataExportManager,
    initial_conditions: ImagesDataset,
    simulation: Simulation,
    params_provider: Callable[[int], th.Tensor],
    rounds: int,
    sim_observations: int,
    max_sim_steps: int,
    min_sim_steps: int = 1,
    batch_size=64,
) -> None:
    images_seq_shape = (sim_observations + 1, *initial_conditions[0].shape)
    time_shape = (sim_observations + 1,)
    params_shape = params_provider(1).shape[1:]

    ics = DataLoader(initial_conditions, batch_size, shuffle=False)
    total_iterations = len(ics) * rounds

    with export.get() as e, tqdm(total=total_iterations) as progress_bar:
        images = e.create_export(DataKey.IMAGES, images_seq_shape)
        times = e.create_export(DataKey.TIMES, time_shape)
        params = e.create_export(DataKey.PARAMS, params_shape)

        for u, _ in tqdm(product(ics, range(rounds)), "Simulation"):
            us = [u]
            ts = [th.zeros((len(u),))]
            p = params_provider(len(u))

            for _ in range(sim_observations):
                steps = th.randint(min_sim_steps, max_sim_steps, size=(len(u),))
                ts.append(ts[-1] + steps)

                with th.no_grad():
                    us.append(simulation(us[-1], p, steps))

            images.export(th.stack(us, 1))
            times.export(th.stack(ts, 1))
            params.export(p)

            progress_bar.update()
