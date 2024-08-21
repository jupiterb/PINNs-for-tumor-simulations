import torch as th
import torch.nn as nn

from typing import Sequence

from phynn.models.base import OptimizerParams
from phynn.models.physics.base import BasePhysicsModel, StepInfo
from phynn.physics import Simulation


class InverseProblemModel(BasePhysicsModel):
    def __init__(
        self,
        simulation: Simulation,
        params_names: Sequence[str],
        optimizer_params: OptimizerParams,
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters()
        self._simulation = simulation
        self._num_params = len(params_names)
        self._params_names = params_names
        self._params = nn.Parameter(th.ones((self._num_params,)))

    @property
    def params(self) -> Sequence[float]:
        return [p.item() for p in self._params.detach()]

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        loss = super().training_step(batch, batch_idx)
        self.log_dict({name: p for name, p in zip(self._params_names, self.params)})
        return loss

    def _step(self, batch: th.Tensor) -> StepInfo:
        u_input, u_target, duration = batch
        params = th.cat(
            [self._params.view((1, self._num_params)) for _ in range(len(duration))]
        )
        u_prediction = self._simulation(u_input, params, duration)
        loss = self._loss(u_prediction, u_target)
        return StepInfo(loss, u_input, u_target, u_prediction)
