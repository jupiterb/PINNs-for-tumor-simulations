from __future__ import annotations

import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Sequence

from phynn.nn.conv import ConvNet, ConvNetParams, ConvNetCommonParams
from phynn.nn.fc import FC, FCParams


@dataclass
class ConvAutoEncoderParams:
    in_shape: Sequence[int]
    conv_encoder_params: ConvNetParams
    fc_encoder_params: FCParams


class ConvAutoEncoder(nn.Module):
    def __init__(self, params: ConvAutoEncoderParams) -> None:
        super(ConvAutoEncoder, self).__init__()

        conv_encoder = ConvNet(params.conv_encoder_params)
        fc_encoder = FC(params.fc_encoder_params)
        self._encoder = nn.Sequential(conv_encoder, nn.Flatten(), fc_encoder)

        with th.no_grad():
            pre_flatten_shape = conv_encoder(th.zeros((1, *params.in_shape))).shape[1:]

        fc_decoder_params = FCParams(
            list(reversed(params.fc_encoder_params.features)),
            list(reversed(params.fc_encoder_params.activations)),
        )
        conv_decoder_params = ConvNetParams(
            list(reversed(params.conv_encoder_params.channels)),
            list(reversed(params.conv_encoder_params.blocks)),
            ConvNetCommonParams(True, True, True),
        )

        fc_decoder = FC(fc_decoder_params)
        unflatten = nn.Unflatten(1, pre_flatten_shape)
        conv_decoder = ConvNet(conv_decoder_params)
        self._decoder = nn.Sequential(fc_decoder, unflatten, conv_decoder)

        self._in_shape = params.in_shape

    @property
    def in_shape(self) -> Sequence[int]:
        return self._in_shape

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def forward(self, x: th.Tensor) -> th.Tensor:
        latent = self._encoder(x)
        return self._decoder(latent)


@dataclass
class VariationalAutoEncoderParams:
    conv_ae_params: ConvAutoEncoderParams
    latent_size: int


class _VariationalEncoder(nn.Module):
    def __init__(
        self, encoder: nn.Module, pre_latent_size: int, latent_size: int
    ) -> None:
        super(_VariationalEncoder, self).__init__()
        self._encoder = encoder
        self._fc_mu = nn.Linear(pre_latent_size, latent_size)
        self._fc_var = nn.Linear(pre_latent_size, latent_size)

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        x = self._encoder(x)
        return self._fc_mu(x), self._fc_var(x)


class _VariationalDecoder(nn.Module):
    def __init__(
        self, decoder: nn.Module, pre_latent_size: int, latent_size: int
    ) -> None:
        super(_VariationalDecoder, self).__init__()
        pre_decoder_fc_params = FCParams([latent_size, pre_latent_size], [nn.LeakyReLU])
        pre_decoder_fc = FC(pre_decoder_fc_params)
        self._decoder = nn.Sequential(pre_decoder_fc, decoder)

    def forward(self, mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        std = th.exp(0.5 * log_var)
        latent = mu + std * th.randn(mu.size()).to(mu.device)
        return self._decoder(latent)


class VariationalAutoEncoder(ConvAutoEncoder):
    def __init__(self, params: VariationalAutoEncoderParams) -> None:
        super().__init__(params.conv_ae_params)

        pre_latent_size = params.conv_ae_params.fc_encoder_params.features[-1]

        self._encoder = _VariationalEncoder(
            self._encoder, pre_latent_size, params.latent_size
        )
        self._decoder = _VariationalDecoder(
            self._decoder, pre_latent_size, params.latent_size
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        mu, log_var = self._encoder(x)
        return self.decoder(mu, log_var)
