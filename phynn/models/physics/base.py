import torch as th
import torch.nn.functional as F

from abc import ABC, abstractmethod
from dataclasses import dataclass

from phynn.models.base import BaseModel


@dataclass
class StepInfo:
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


class BasePhysicsModel(BaseModel, ABC):
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
    def _step(self, batch: th.Tensor) -> StepInfo:
        raise NotImplementedError()

    def _loss(self, prediction: th.Tensor, target: th.Tensor) -> th.Tensor:
        return F.mse_loss(prediction, target)
