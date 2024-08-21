import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Sequence, Type

Activation = Type[nn.Module]


@dataclass
class FCParams:
    features: Sequence[int]
    activations: Sequence[Activation]


class FC(nn.Module):
    def __init__(self, params: FCParams) -> None:
        super(FC, self).__init__()
        self._fc = nn.Sequential()
        in_features = params.features[0]

        for out_features, activation in zip(params.features[1:], params.activations):
            self._fc.append(nn.Linear(in_features, out_features))
            self._fc.append(activation())
            in_features = out_features

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._fc(x)
