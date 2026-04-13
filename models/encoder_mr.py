# 3D feature extraction from MR volumes using 3D ResNet-18.

import torch
import torch.nn as nn
import torchvision.models.video as video_models

class MREncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet3d = video_models.r3d_18(weights=None)
        # Modifying the first layer for 1-channel grayscale MR input
        resnet3d.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        
        num_ftrs = resnet3d.fc.in_features
        resnet3d.fc = nn.Linear(num_ftrs, embedding_dim)
        self.model = resnet3d

    def forward(self, x):
        return self.model(x)