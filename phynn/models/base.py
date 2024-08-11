import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch as th
from torch import optim
from torchvision.utils import make_grid

from dataclasses import dataclass

from typing import Type


@dataclass
class OptimizerParams:
    optimizer: (
        Type[optim.Adam] | Type[optim.AdamW] | Type[optim.RMSprop] | Type[optim.SGD]
    )
    lr: float


class BaseModel(L.LightningModule):
    def __init__(self, optimizer_params: OptimizerParams) -> None:
        super().__init__()
        self._optimizer_params = optimizer_params

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = self._optimizer_params.optimizer(
            self.parameters(), lr=self._optimizer_params.lr
        )
        return optimizer

    def _log_visualize(self, key: str, tensors: dict[str, th.Tensor]) -> None:
        if isinstance(self.logger, WandbLogger):
            grids = [make_grid(t, 1) for t in tensors.values()]
            names = [n for n in tensors]
            self.logger.log_image(key, grids, caption=names)
