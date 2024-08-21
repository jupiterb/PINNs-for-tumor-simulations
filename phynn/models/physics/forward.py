import torch as th

from dataclasses import dataclass
from typing import Sequence

from phynn.models.base import OptimizerParams
from phynn.models.physics.base import BasePhysicsModel, StepInfo
from phynn.physics import Simulation


@dataclass
class ForwardStepInfo(StepInfo):
    u_pde_residuum: th.Tensor

    def u_asdict(self) -> dict[str, th.Tensor]:
        d = super().u_asdict()
        d["u_pde_residuum"] = self.u_pde_residuum
        return d


class ForwardProblemModel(BasePhysicsModel):
    def __init__(
        self,
        simulation: Simulation,
        params: Sequence[float],
        optimizer_params: OptimizerParams,
        pde_residuum_weight: float = 0.5,
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters()
        self._simulation = simulation
        self._params = th.Tensor(params)
        self._num_params = len(params)
        self._pde_weight = pde_residuum_weight

    def _step(self, batch: th.Tensor) -> StepInfo:
        params = self._params.view((1, self._num_params))
        u_input, u_target, u_pde_residuum, duration = batch
        u_prediction = self._simulation(u_input, params, duration)
        loss_data = self._loss(u_prediction, u_target)
        loss_pde = self._loss(u_prediction, u_pde_residuum)
        loss = (1 - self._pde_weight) * loss_data + self._pde_weight * loss_pde
        return ForwardStepInfo(loss, u_input, u_target, u_prediction, u_pde_residuum)
