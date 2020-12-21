import torch
import torch.nn as nn


class WeightedRegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask, fg_weight=1.0, bg_weight=0.1):
        assert pred.size() == gt.size()
        mask = mask[:, None, :, :].expand_as(pred)
        weight = torch.where(
            torch.eq(mask, 0.5), torch.tensor([fg_weight]).cuda(), torch.tensor([bg_weight]).cuda())
        loss = ((pred - gt)**2) * weight

        loss = loss.mean() * 1000
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


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1))
        heatmaps_gt = target.reshape(
            (batch_size, num_joints, -1))
        weights = torch.where(torch.gt(heatmaps_gt, 0.1), torch.tensor(
            [1.0]).cuda(), torch.tensor([0.1]).cuda())
        
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[:, idx]
            heatmap_gt = heatmaps_gt[:,idx]
            weight = weights[:,idx]
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(weight),
                                       heatmap_gt.mul(weight))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss
