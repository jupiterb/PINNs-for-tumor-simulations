import torch as th
from torch.utils.data import Dataset
from typing import Callable, Sequence

from phynn.data.img.interface import ImagesDataInterface
from phynn.physics import ParametrizedEquation, EquationSimulation


SimulationDataSample = tuple[
    th.Tensor,  # simulation start
    th.Tensor,  # simulation result
    th.Tensor,  # params
    th.Tensor,  # duration
]


class DynamicSimulationDataset(Dataset):
    def __init__(
        self,
        initial_conditions: ImagesDataInterface,
        diff_eq: ParametrizedEquation,
        params_provider: Callable[[int], th.Tensor],
        max_simulation_steps: int,
        min_simulation_steps: int = 1,
        max_pre_steps: int = 0,
    ) -> None:
        super().__init__()
        self._ics = initial_conditions
        self._simulation = EquationSimulation(diff_eq)
        self._params = params_provider
        self._max_sim_steps = max_simulation_steps
        self._min_sim_steps = min_simulation_steps
        self._max_pre_steps = max_pre_steps

    def __len__(self) -> int:
        return self._ics.size

    def __getitem__(self, index: int) -> SimulationDataSample:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[SimulationDataSample]:
        batch_size = len(indices)

        ics = self._ics.get_batch(indices)
        params = self._params(len(indices))
        pre_steps = th.randint(low=0, high=self._max_pre_steps, size=(batch_size,))
        sim_steps = th.randint(
            low=self._min_sim_steps, high=self._max_sim_steps, size=(batch_size,)
        )

        with th.no_grad():
            starts = self._simulation(ics, params, pre_steps)
            results = self._simulation(starts, params, sim_steps)

        return [
            (starts[i], results[i], params[i], sim_steps[i]) for i in range(batch_size)
        ]
