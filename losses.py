import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        preds,
        targets,
        smooth=1
    ):

        preds = torch.sigmoid(preds)

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (
            preds * targets
        ).sum()

        dice = (
            2 * intersection + smooth
        ) / (
            preds.sum() +
            targets.sum() +
            smooth
        )

        return 1 - dice


class HybridLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, preds, targets):

        bce_loss = self.bce(
            preds,
            targets
        )

        dice_loss = self.dice(
            preds,
            targets
        )

        return (
            0.5 * bce_loss +
            0.5 * dice_loss
        )