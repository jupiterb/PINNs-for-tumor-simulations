import torch as th
import torch.nn as nn

from typing import Sequence

from phynn.data import SequenceImagesDataset
from phynn.experiments.base_physics import BasePhysicsExperiment
from phynn.experiments.utils import StageConfig, run_stage
from phynn.models import VAEModel
from phynn.nn import FC, FCParams, VariationalAutoEncoder, VariationalAutoEncoderParams
from phynn.physics import Simulation


class Frozen(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super(Frozen, self).__init__()
        self._net = net

    def forward(self, *x):
        with th.no_grad():
            return self._net(*x)


class VAELatentProcessingSimulation(Simulation):
    def __init__(
        self,
        vae: VariationalAutoEncoder,
        processor: FC,
        froze_encoder: bool = False,
        froze_processor: bool = False,
        froze_decoder: bool = False,
    ) -> None:
        super(VAELatentProcessingSimulation, self).__init__()
        self._encoder = Frozen(vae.encoder) if froze_encoder else vae.encoder
        self._decoder = Frozen(vae.decoder) if froze_decoder else vae.decoder
        self._processor = Frozen(processor) if froze_processor else processor

    def forward(self, u: th.Tensor, params: th.Tensor, t: th.Tensor) -> th.Tensor:
        mu, log_var = self._encoder(u)
        latent = th.cat((mu, params, t), dim=1)
        latent_processed = self._processor(latent)
        zeros_log_var = th.zeros_like(log_var)
        return self._decoder(latent_processed, zeros_log_var)


class VAELatentProcessingExperiment(BasePhysicsExperiment):
    def __init__(
        self,
        vae_params: VariationalAutoEncoderParams,
        processor_params: FCParams,
        params_names: Sequence[str],
        vae_model_stage_config: StageConfig | None,
        general_model_stage_config: StageConfig | None,
        inverse_model_stage_config: StageConfig | None,
        forward_model_stage_config: StageConfig | None,
    ) -> None:
        self._vae = VariationalAutoEncoder(vae_params)
        self._processor = FC(processor_params)
        self._vae_model_stage_config = vae_model_stage_config
        super().__init__(
            params_names,
            general_model_stage_config,
            inverse_model_stage_config,
            forward_model_stage_config,
        )

    def run(self) -> None:
        if (config := self._vae_model_stage_config) is not None:
            vae_model = run_stage(
                VAEModel,
                SequenceImagesDataset,
                config,
                vae=self._vae,
                optimizer_params=config.optimizer_params,
            )

        super().run()

    def _get_general_problem_simulation(self) -> Simulation:
        return VAELatentProcessingSimulation(
            self._vae, self._processor, froze_encoder=True, froze_decoder=True
        )

    def _get_inverse_problem_simulation(self) -> Simulation:
        return VAELatentProcessingSimulation(
            self._vae,
            self._processor,
            froze_encoder=True,
            froze_decoder=True,
            froze_processor=True,
        )

    def _get_forward_problem_simulation(self) -> Simulation:
        return VAELatentProcessingSimulation(self._vae, self._processor)
