import torch as th
import torch.nn.functional as F

from dataclasses import dataclass

from phynn.models.base import BaseModel, OptimizerParams
from phynn.nn import VariationalAutoEncoder


@dataclass
class _VAELossInfo:
    loss: th.Tensor
    reconstruction_loss: th.Tensor
    kld_loss: th.Tensor

    def asdict(self) -> dict[str, th.Tensor]:
        return {
            "loss": self.loss,
            "reconstruction_loss": self.reconstruction_loss,
            "kld_loss": self.kld_loss,
        }


_Reconstructions = th.Tensor


class VAEModel(BaseModel):
    def __init__(
        self, vae: VariationalAutoEncoder, optimizer_params: OptimizerParams
    ) -> None:
        super().__init__(optimizer_params)
        self._vae = vae
        self._kld_weight = 1 / th.prod(th.Tensor(vae.in_shape))
        self._epochs = 0

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        x = batch
        info, _ = self._step(x)
        self.log_dict(info.asdict())
        return info.loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        x = batch
        info, x_hat = self._step(x)
        self.log_dict({"val_" + name: value for name, value in info.asdict().items()})

        if batch_idx == 0:
            self._maybe_log_in_vs_out(x, x_hat)

        return info.loss

    def _maybe_log_in_vs_out(self, x: th.Tensor, x_hat: th.Tensor) -> None:
        if self._epochs % 10 == 0:
            self._log_visualize(
                "in_vs_out_visualization",
                {"inputs": x, "reconstructions": x_hat},
            )

        self._epochs += 1

    def _step(self, x: th.Tensor) -> tuple[_VAELossInfo, _Reconstructions]:
        mu, log_var = self._vae.encoder(x)
        x_hat = self._vae.decoder(mu, log_var)

        reconstruction_loss = self._reconstruction_loss(x, x_hat)
        kld_loss = self._kld_loss(mu, log_var)
        loss = reconstruction_loss + kld_loss

        return _VAELossInfo(loss, reconstruction_loss, kld_loss), x_hat

    def _reconstruction_loss(self, x: th.Tensor, x_hat: th.Tensor) -> th.Tensor:
        return F.mse_loss(x_hat, x)

    def _kld_loss(self, mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        var = th.exp(log_var)
        return (
            -0.5
            * self._kld_weight
            * th.mean(th.sum(1 + log_var - var - mu**2, dim=1), dim=0)
        )
