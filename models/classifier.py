import torch
import torch.nn as nn


class XrayClassifier(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, embedding):
        logits = self.classifier(embedding)
        return logits.squeeze(1)