import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallXrayEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.stem = ConvBlock(1, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        vec = self.pool(x4)
        vec = torch.flatten(vec, 1)
        embedding = self.fc(vec)

        return embedding, [x0, x1, x2, x3, x4]
