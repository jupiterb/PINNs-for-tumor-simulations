import torch.nn as nn

from typing import Type

from phynn.nn.base import SequentialNetCreator, SequentialNetParams


Activation = Type[nn.Module]


class FC(SequentialNetCreator[int, Activation]):
    def create(self, params: SequentialNetParams[int, Activation]) -> nn.Module:
        fc = nn.Sequential()
        in_features = params.in_space

        for out_features, activation in params:
            fc += FC._create_block(in_features, out_features, activation)
            in_features = out_features

        return fc

    @staticmethod
    def _create_block(
        in_features: int, out_features: int, activation: Activation
    ) -> nn.Sequential:
        block = nn.Sequential()
        block.append(nn.Linear(in_features, out_features))
        block.append(activation())
        return block
