import torch as th
import torch.nn as nn

from abc import ABC, abstractmethod


class Simulation(nn.Module, ABC):
    def __init__(self) -> None:
        super(Simulation, self).__init__()

    @abstractmethod
    def forward(self, u: th.Tensor, params: th.Tensor, t: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


class DiscreteSimulationStep(nn.Module, ABC):
    def __init__(self) -> None:
        super(DiscreteSimulationStep, self).__init__()

    @abstractmethod
    def forward(self, u: th.Tensor, params: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


class DiscreteSimulation(Simulation):
    def __init__(
        self,
        simulation_step: DiscreteSimulationStep,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> None:
        super(DiscreteSimulation, self).__init__()
        self._step = simulation_step
        self._clip = lambda u: th.clip(u, min_value, max_value)

    def forward(self, u: th.Tensor, params: th.Tensor, t: th.Tensor) -> th.Tensor:
        max_duration = int(t.max().item())
        u = u.clone()

        for step in range(max_duration):
            mask = t > step
            mask = mask.squeeze() if mask.dim() > 1 else mask

            diff = self._step(u[mask], params[mask])

            u[mask] += diff
            u = self._clip(u)

        return u
