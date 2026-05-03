# Model loss calculations

import torch
import torch.nn as nn

class HybridLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight_val=5.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight_val], dtype=torch.float32))

    def forward(self, pred, target):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=self.pos_weight
        )

        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-6

        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth))
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class LatentAlignmentLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, embedding1, embedding2, target_labels=None):
        if target_labels is None:
            target_labels = torch.ones(embedding1.size(0)).to(embedding1.device)
            
        return self.loss_fn(embedding1, embedding2, target_labels)