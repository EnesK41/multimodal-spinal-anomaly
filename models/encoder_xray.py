# Feature extraction from 2D X-ray using ResNet-50.

import torch
import torch.nn as nn
import torchvision.models as models

class XrayEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet = models.resnet34(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, embedding_dim)
        self.model = resnet

    def forward(self, x):
        return self.model(x)