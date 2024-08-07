import torch.nn as nn

from phynn.nn import Conv, ConvBlockParams, get_factory, UNet, UNetLevelParams
from phynn.train import training_device


def create_u_net(
    num_levels: int = 3, initial_channels: int = 64, out_channels: int = 2
) -> nn.Module:
    encoder_creator = Conv(rescale_on_begin=True)
    decoder_creator = Conv(upsample=True)

    unet_creator = UNet(encoder_creator, decoder_creator)
    unet_factory = get_factory(unet_creator)
    unet_params = unet_factory.init(initial_channels)

    level_channels = initial_channels

    for _ in range(num_levels):
        unet_params += unet_factory.layer(
            level_channels, UNetLevelParams(level_channels * 2, ConvBlockParams())
        ) + unet_factory.layer(level_channels * 2, ConvBlockParams(rescale=2))

        level_channels *= 2

    unet = unet_creator.create(unet_params)

    conv_creator = Conv()
    conv_factory = get_factory(conv_creator)

    unet_in = conv_creator.create(
        conv_factory.init(1) + conv_factory.layer(initial_channels, ConvBlockParams())
    )

    unet_out = conv_creator.create(
        conv_factory.init(initial_channels)
        + conv_factory.layer(initial_channels, ConvBlockParams())
        + conv_factory.layer(out_channels, ConvBlockParams(1, activation=nn.Hardtanh))
    )

    return nn.Sequential(unet_in, unet, unet_out).to(training_device)
