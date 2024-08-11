import torch as th
import torch.nn as nn

from typing import Sequence


class EquationComponents(nn.Module):
    def __init__(self, neural_eq_components: Sequence[nn.Module]) -> None:
        super(EquationComponents, self).__init__()
        self._nns = nn.ModuleList(neural_eq_components)

    def forward(self, u: th.Tensor) -> th.Tensor:
        return th.cat([nn(u) for nn in self._nns], 1)


class ParametrizedEquation(nn.Module):
    def __init__(self, diff_eq_components: nn.Module) -> None:
        super(ParametrizedEquation, self).__init__()
        self._diff_eq_components = diff_eq_components

    def _components(self, u: th.Tensor) -> th.Tensor:
        return self._diff_eq_components(u)

    def forward(self, u: th.Tensor, params: th.Tensor) -> th.Tensor:
        components = self._components(u)
        expand_dims = params.shape + (1,) * (u.ndim - 2)
        return (components * params.view(expand_dims)).sum(1, keepdim=True)


class FrozenParametrizedEquation(ParametrizedEquation):
    def __init__(self, diff_eq_components: nn.Module) -> None:
        super().__init__(diff_eq_components)

    def _components(self, u: th.Tensor) -> th.Tensor:
        with th.no_grad():
            return super()._components(u)


class EquationSimulation(nn.Module):
    def __init__(
        self,
        equation: ParametrizedEquation,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> None:
        super(EquationSimulation, self).__init__()
        self._equation = equation
        self._clip = lambda u: th.clip(u, min_value, max_value)

    def forward(self, u: th.Tensor, params: th.Tensor, t: th.Tensor) -> th.Tensor:
        max_duration = int(t.max().item())
        u = u.clone()

        for step in range(max_duration):
            mask = t > step
            mask = mask.squeeze() if mask.dim() > 1 else mask

            diff = self._equation(u[mask], params[mask])

            u[mask] += diff
            u = self._clip(u)

        return u
