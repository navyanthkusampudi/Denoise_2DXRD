import torch.nn as nn
from typing import Optional


class DnCNN(nn.Module):
    """
    DnCNN implementation.

    This repo historically used the signature:
        DnCNN(channels=..., num_of_layers=...)

    Some training/inference entrypoints also call:
        DnCNN(in_channels, out_channels)

    To keep both working, we accept either:
      - channels=K  (implies in_channels=out_channels=K)
      - (in_channels, out_channels) positional
      - in_channels=..., out_channels=...
    """

    def __init__(
        self,
        channels: Optional[int] = None,
        num_of_layers: int = 17,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super(DnCNN, self).__init__()
        # Back-compat: allow legacy kwargs or unexpected extras without crashing.
        _ = args, kwargs

        if in_channels is None and out_channels is None:
            if channels is None:
                raise TypeError("DnCNN requires either channels=... or in_channels/out_channels.")
            in_channels = int(channels)
            out_channels = int(channels)
        else:
            if in_channels is None or out_channels is None:
                raise TypeError("DnCNN requires both in_channels and out_channels (or channels=...).")
            in_channels = int(in_channels)
            out_channels = int(out_channels)

        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
