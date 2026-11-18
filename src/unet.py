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

    def __init__(self, dropout=0.5):
        super().__init__()

        self.conv = DoubleConv(512, 1024)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)

        return x

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

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels * 2,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        skip = center_crop(skip, x)

        x = torch.cat([skip, x], dim=1)

        x = self.conv(x)

        return x

class OutputLayer(nn.Module):

    def __init__(self, in_channels=64, num_classes=2):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        # Encoder
        self.encoder = Encoder(in_channels)

        # Bottleneck
        self.bottleneck = Bottleneck()

        # Decoder
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # Output layer
        self.output = OutputLayer(64, num_classes)

    def forward(self, x):

        # Contracting path
        x, (e1, e2, e3, e4) = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Expanding path
        x = self.dec4(x, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)

        # Segmentation logits
        x = self.output(x)

        return x

