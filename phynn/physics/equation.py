import torch as th
import torch.nn as nn

from typing import Sequence

from phynn.physics.simulation import DiscreteSimulationStep


class EquationComponents(nn.Module):
    def __init__(self, equation_components_list: Sequence[nn.Module]) -> None:
        super(EquationComponents, self).__init__()
        self._components = nn.ModuleList(equation_components_list)

    def forward(self, u: th.Tensor) -> th.Tensor:
        return th.cat([c(u) for c in self._components], 1)


class LinearEquation(DiscreteSimulationStep):
    def __init__(self, equation_components: nn.Module) -> None:
        super(LinearEquation, self).__init__()
        self._equation_components = equation_components

    def _components(self, u: th.Tensor) -> th.Tensor:
        return self._equation_components(u)

    def forward(self, u: th.Tensor, params: th.Tensor) -> th.Tensor:
        components = self._components(u)
        expand_dims = params.shape + (1,) * (u.ndim - 2)
        return (components * params.view(expand_dims)).sum(1, keepdim=True)


class FrozenLinearEquation(LinearEquation):
    def __init__(self, diff_eq_components: nn.Module) -> None:
        super().__init__(diff_eq_components)

    def _components(self, u: th.Tensor) -> th.Tensor:
        with th.no_grad():
            return super()._components(u)
