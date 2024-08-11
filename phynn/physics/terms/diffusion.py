import torch as th
import torch.nn as nn


class DiffusionTerm(nn.Module):
    def __init__(self) -> None:
        super(DiffusionTerm, self).__init__()

    def forward(self, u: th.Tensor) -> th.Tensor:
        diffusion = th.zeros_like(u)

        for dim in range(2, u.dim()):
            diffusion += th.gradient(th.gradient(u, dim=dim)[0], dim=dim)[0]

        return diffusion
