import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class MaskDecoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.ag1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, embedding, skip_features):
        x0, x1, x2, x3, x4 = skip_features
        
        emb = embedding.view(-1, 512, 1, 1)
        x4_conditioned = x4 * emb
        
        d4 = self.upconv4(x4_conditioned)
        x3_att = self.ag4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.dec_conv4(d4)
        
        d3 = self.upconv3(d4)
        x2_att = self.ag3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.dec_conv3(d3)
        
        d2 = self.upconv2(d3)
        x1_att = self.ag2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.dec_conv2(d2)
        
        d1 = self.upconv1(d2)
        x0_att = self.ag1(g=d1, x=x0)
        d1 = torch.cat((x0_att, d1), dim=1)
        d1 = self.dec_conv1(d1)

        d_final = self.final_upconv(d1)
        out = self.final_conv(d_final)
        
        return out