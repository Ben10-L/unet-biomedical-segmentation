import torch.nn as nn


def initialize_weights(module):

    if isinstance(module, nn.Conv2d):

        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_in",
            nonlinearity="relu"
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.ConvTranspose2d):

        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu"
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)