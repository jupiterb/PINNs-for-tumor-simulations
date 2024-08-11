import torch as th
import torch.nn as nn


class ProliferationTerm(nn.Module):
    def __init__(self) -> None:
        super(ProliferationTerm, self).__init__()

    def forward(self, u: th.Tensor) -> th.Tensor:
        return u * (1 - u)
