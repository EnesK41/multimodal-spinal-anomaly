# Model loss calculations

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class LatentAlignmentLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, embedding1, embedding2, target_labels):
        return self.loss_fn(embedding1, embedding2, target_labels)