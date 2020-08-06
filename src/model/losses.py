import torch
import torch.nn as nn


class WeightedRegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask, fg_weight=1.0, bg_weight=0.1):
        assert pred.size() == gt.size()
        mask = mask[:, None, :, :].expand_as(pred)
        weight = torch.where(
            torch.eq(mask, 1), torch.tensor([fg_weight]).cuda(), torch.tensor([bg_weight]).cuda())
        loss = ((pred - gt)**2) * weight

        loss = loss.mean()
        return loss

class RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        mask = mask[:, None, :, :].expand_as(pred)
        loss = ((pred - gt)**2) * mask[:, None, :, :]

        if vis_map is not None:
            loss = loss * vis_map

        loss = loss.mean()
        return loss
