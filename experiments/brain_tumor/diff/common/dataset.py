import torch as th
import torch.nn as nn
import torchvision.transforms.v2 as T

from pathlib import Path
from typing import Sequence

from phynn.data.img import (
    ImagesDataInterface,
    ImagesDataInterfaceWrapper,
    HDF5ImagesDataInterface,
    train_test_split,
)
from phynn.data.sim import DynamicSimulationDataset
from phynn.physics import EquationComponents, ParametrizedEquation
from phynn.physics.terms import DiffusionTerm, ProliferationTerm
from phynn.train import training_device


def get_data() -> tuple[ImagesDataInterface, ImagesDataInterface]:
    path = Path("./../data/processed/BRATS2020/result.h5")

    raw = HDF5ImagesDataInterface(path, training_device)

    vertical_flip = T.RandomVerticalFlip()
    horizontal_flip = T.RandomHorizontalFlip()
    rotate = T.RandomRotation(degrees=90)  # type: ignore
    move = T.RandomAffine(degrees=0, translate=(0.1, 0.1))  # type: ignore

    def augmentation(u: th.Tensor) -> th.Tensor:
        u = vertical_flip(u)
        u = horizontal_flip(u)
        u = rotate(u)
        u = move(u)
        return u

    augmented = ImagesDataInterfaceWrapper(raw, augmentation)

    return train_test_split(augmented, 0.7)


class AugmentedDiffEquation(ParametrizedEquation):
    def __init__(
        self, neural_eq_components: Sequence[nn.Module], threshold: float = 0.2
    ) -> None:
        diff_eq_components_net = EquationComponents(neural_eq_components)
        super().__init__(diff_eq_components_net)
        self._threshold = threshold

    def forward(self, u: th.Tensor, params: th.Tensor) -> th.Tensor:
        # change params a little
        params = params * (1 + (th.rand_like(params) - 0.5) / 10)

        # apply only on tumor
        v = u - self._threshold
        v[v < 0] = 0
        v /= 1 - self._threshold
        diff = super().forward(v, params)
        diff[u < 0.05] = 0

        return diff


def create_dataset(
    initial_conditions: ImagesDataInterface,
    max_simulation_steps: int = 8,
    min_simulation_steps: int = 5,
    max_pre_steps: int = 3,
) -> DynamicSimulationDataset:
    diff_eq = AugmentedDiffEquation([DiffusionTerm(), ProliferationTerm()])
    diff_eq = diff_eq.to(training_device)

    params = lambda batch_size: (
        (th.rand((batch_size, 2)) * th.Tensor([[2.0, 2.0]]) + th.Tensor([[0.5, 0.5]]))
        ** 2.0
    ).to(training_device)

    return DynamicSimulationDataset(
        initial_conditions=initial_conditions,
        diff_eq=diff_eq,
        params_provider=params,
        max_simulation_steps=max_simulation_steps,
        min_simulation_steps=min_simulation_steps,
        max_pre_steps=max_pre_steps,
    )
