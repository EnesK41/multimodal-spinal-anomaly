# Anomaly segmentation mask generation from embeddings using UNet Decoder.

import torch
import torch.nn as nn

class MaskDecoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        # Project 512 vector back to a spatial feature map of size 256 x 8 x 8
        self.fc = nn.Linear(embedding_dim, 256 * 8 * 8)
        
        # Upsampling layers to go from 8x8 to 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)     # 256x256
        )

    def forward(self, x):
        # x is the [batch_size, 512] vector
        x = self.fc(x)
        # Reshape to [batch_size, channels, height, width]
        x = x.view(x.size(0), 256, 8, 8)
        # Upsample to full resolution mask
        x = self.decoder(x)
        return x