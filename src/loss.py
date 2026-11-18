import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, weight_map):

        pixel_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        weighted_loss = pixel_loss * weight_map

        return weighted_loss.mean()