from abc import ABC, abstractmethod
from typing import Sequence

from phynn.data import (
    SimulationSamplesDataset,
    SequenceSamplesDataset,
    PhysicsResiduumsSamplesDataset,
)
from phynn.experiments.utils import StageConfig, run_stage
from phynn.models import (
    GeneralPhysicsModel,
    InverseProblemModel,
    ForwardProblemModel,
)
from phynn.nn import UNet, UNetParams
from phynn.physics import (
    DiscreteSimulation,
    LinearEquation,
    FrozenLinearEquation,
    Simulation,
)


class BasePhysicsExperiment(ABC):
    def __init__(
        self,
        params_names: Sequence[str],
        general_model_stage_config: StageConfig | None,
        inverse_model_stage_config: StageConfig | None,
        forward_model_stage_config: StageConfig | None,
    ) -> None:
        self._params_names = params_names

        self._general_model_stage_config = general_model_stage_config
        self._inverse_model_stage_config = inverse_model_stage_config
        self._forward_model_stage_config = forward_model_stage_config

    def run(self) -> None:
        if (config := self._general_model_stage_config) is not None:
            general_model = run_stage(
                GeneralPhysicsModel,
                SimulationSamplesDataset,
                config,
                simulation=self._get_general_problem_simulation(),
                optimizer_params=config.optimizer_params,
            )

        inverse_model = None

        if (config := self._inverse_model_stage_config) is not None:
            inverse_model = run_stage(
                InverseProblemModel,
                SequenceSamplesDataset,
                config,
                simulation=self._get_inverse_problem_simulation(),
                params_names=self._params_names,
                optimizer_params=config.optimizer_params,
            )

        if (config := self._forward_model_stage_config) is not None:
            if inverse_model is None:
                raise ValueError(
                    "Cannot run Forward Problem Stage if Inverse Problem Stage is uncompleted."
                )

            forward_model = run_stage(
                ForwardProblemModel,
                PhysicsResiduumsSamplesDataset,
                config,
                simulation=self._get_forward_problem_simulation(),
                params=inverse_model.params,
                optimizer_params=config.optimizer_params,
            )

    @abstractmethod
    def _get_general_problem_simulation(self) -> Simulation:
        raise NotImplementedError()

    @abstractmethod
    def _get_inverse_problem_simulation(self) -> Simulation:
        raise NotImplementedError()

    @abstractmethod
    def _get_forward_problem_simulation(self) -> Simulation:
        raise NotImplementedError()
