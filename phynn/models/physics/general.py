import torch as th

from phynn.models.base import OptimizerParams
from phynn.models.physics.base import BasePhysicsModel, StepInfo
from phynn.physics import Simulation


class GeneralPhysicsModel(BasePhysicsModel):
    def __init__(
        self, simulation: Simulation, optimizer_params: OptimizerParams
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters()
        self._simulation = simulation

    def _step(self, batch: th.Tensor) -> StepInfo:
        u_input, u_target, duration, params = batch
        u_prediction = self._simulation(u_input, params, duration)
        loss = self._loss(u_prediction, u_target)
        return StepInfo(loss, u_input, u_target, u_prediction)
