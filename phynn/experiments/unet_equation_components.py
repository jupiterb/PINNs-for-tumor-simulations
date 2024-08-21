from typing import Sequence

from phynn.experiments.base_physics import BasePhysicsExperiment
from phynn.experiments.utils import StageConfig
from phynn.nn import UNet, UNetParams
from phynn.physics import (
    DiscreteSimulation,
    LinearEquation,
    FrozenLinearEquation,
    Simulation,
)


class UNetEquationComponentsExperiment(BasePhysicsExperiment):
    def __init__(
        self,
        unet_params: UNetParams,
        params_names: Sequence[str],
        general_model_stage_config: StageConfig | None,
        inverse_model_stage_config: StageConfig | None,
        forward_model_stage_config: StageConfig | None,
    ) -> None:
        self._unet = UNet(unet_params)
        super().__init__(
            params_names,
            general_model_stage_config,
            inverse_model_stage_config,
            forward_model_stage_config,
        )

    def _get_general_problem_simulation(self) -> Simulation:
        return self._get_simulation()

    def _get_inverse_problem_simulation(self) -> Simulation:
        return self._get_frozen_simulation()

    def _get_forward_problem_simulation(self) -> Simulation:
        return self._get_simulation()

    def _get_simulation(self) -> Simulation:
        equation = LinearEquation(self._unet)
        return DiscreteSimulation(equation)

    def _get_frozen_simulation(self) -> Simulation:
        equation = FrozenLinearEquation(self._unet)
        return DiscreteSimulation(equation)
