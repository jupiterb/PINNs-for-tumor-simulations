import torch.nn as nn
import torch.optim as optim

from phynn.data.sim import DynamicSimulationDataset
from phynn.physics import ParametrizedEquation, EquationSimulation
from phynn.models import GeneralDiffEquationModel, OptimizerParams
from phynn.train import train


def run_training(
    diff_eq_components_net: nn.Module,
    train_ds: DynamicSimulationDataset,
    test_ds: DynamicSimulationDataset,
    run_name: str,
    batch_size: int,
    epochs: int,
    lr: float = 0.00005,
) -> None:
    diff_eq_nn = ParametrizedEquation(diff_eq_components_net)
    diff_simulation = EquationSimulation(diff_eq_nn)
    diff_eq_model = GeneralDiffEquationModel(
        diff_simulation, optimizer_params=OptimizerParams(optim.AdamW, lr)
    )

    train(
        diff_eq_model,
        run_name=run_name,
        train_dataset=train_ds,
        val_dataset=test_ds,
        batch_size=batch_size,
        epochs=epochs,
    )
