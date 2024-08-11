import torch as th
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from phynn.models.base import BaseModel, OptimizerParams
from phynn.physics import EquationSimulation


@dataclass
class _StepInfo:
    loss: th.Tensor
    u_input: th.Tensor
    u_target: th.Tensor
    u_prediction: th.Tensor

    def u_asdict(self) -> dict[str, th.Tensor]:
        return {
            "u_input": self.u_input,
            "u_target": self.u_target,
            "u_prediction": self.u_prediction,
        }


class BaseDiffEquationModel(BaseModel, ABC):
    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        info = self._step(batch)
        self.log_dict({"loss": info.loss})
        return info.loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        info = self._step(batch)
        self.log_dict({"val_loss": info.loss})

        if batch_idx == 0:
            self._log_visualize("val_visualization", info.u_asdict())

        return info.loss

    @abstractmethod
    def _step(self, batch: th.Tensor) -> _StepInfo:
        raise NotImplementedError()

    def _loss(self, prediction: th.Tensor, target: th.Tensor) -> th.Tensor:
        return F.mse_loss(prediction, target)


class GeneralDiffEquationModel(BaseDiffEquationModel):
    def __init__(
        self, simulation: EquationSimulation, optimizer_params: OptimizerParams
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters()
        self._simulation = simulation

    def _step(self, batch: th.Tensor) -> _StepInfo:
        u_input, u_target, params, duration = batch
        u_prediction = self._simulation(u_input, params, duration)
        loss = self._loss(u_prediction, u_target)
        return _StepInfo(loss, u_input, u_target, u_prediction)


@dataclass
class _ForwardStepInfo(_StepInfo):
    u_pde_residuum: th.Tensor

    def u_asdict(self) -> dict[str, th.Tensor]:
        d = super().u_asdict()
        d["u_pde_residuum"] = self.u_pde_residuum
        return d


class ForwardProblemDiffEquationModel(BaseDiffEquationModel):
    def __init__(
        self,
        simulation: EquationSimulation,
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

    def _step(self, batch: th.Tensor) -> _StepInfo:
        params = self._params.view((1, self._num_params))
        u_input, u_target, u_pde_residuum, duration = batch
        u_prediction = self._simulation(u_input, params, duration)
        loss_data = self._loss(u_prediction, u_target)
        loss_pde = self._loss(u_prediction, u_pde_residuum)
        loss = (1 - self._pde_weight) * loss_data + self._pde_weight * loss_pde
        return _ForwardStepInfo(loss, u_input, u_target, u_prediction, u_pde_residuum)


class InverseProblemDiffEquationModel(BaseDiffEquationModel):
    def __init__(
        self,
        simulation: EquationSimulation,
        num_params: int,
        optimizer_params: OptimizerParams,
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters()
        self._simulation = simulation
        self._num_params = num_params
        self._params = nn.Parameter(th.empty((num_params,)))

    @property
    def params(self) -> th.Tensor:
        return self._params.detach()

    def _step(self, batch: th.Tensor) -> _StepInfo:
        params = self._params.view((1, self._num_params))
        u_input, u_target, duration = batch
        u_prediction = self._simulation(u_input, params, duration)
        loss = self._loss(u_prediction, u_target)
        return _StepInfo(loss, u_input, u_target, u_prediction)
