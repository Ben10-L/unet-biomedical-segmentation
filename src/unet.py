import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=3, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        x = self.pool(e4)

        return x, (e1, e2, e3, e4)

class Bottleneck(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = DoubleConv(512, 1024)

    def forward(self, x):
        return self.conv(x)

def center_crop(encoder_feature, target_feature):

    h, w = encoder_feature.shape[-2:]

    target_h, target_w = target_feature.shape[-2:]

    if h < target_h or w < target_w:
        raise ValueError("Invalid crop dimensions")

    top = (h - target_h) // 2
    left = (w - target_w) // 2

    return encoder_feature[
        :,
        :,
        top:top + target_h,
        left:left + target_w
    ]
