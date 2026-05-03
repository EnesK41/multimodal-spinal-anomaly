# Feature extraction from 2D X-ray using ResNet-50.

import torch
import torch.nn as nn
import torchvision.models as models

class XrayEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.resnet = models.resnet34(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        # 128x128 boyutundaki ilk katmani (x0) U-Net icin kurtariyoruz
        x0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        x0_pool = self.resnet.maxpool(x0) 

        x1 = self.resnet.layer1(x0_pool)  # 64x64
        x2 = self.resnet.layer2(x1)       # 32x32
        x3 = self.resnet.layer3(x2)       # 16x16
        x4 = self.resnet.layer4(x3)       # 8x8

        vec = self.resnet.avgpool(x4)
        vec = torch.flatten(vec, 1)
        embedding = self.fc(vec)

        # x0 da listeye eklendi (Artik 5 adet atlama baglantimiz var)
        return embedding, [x0, x1, x2, x3, x4]