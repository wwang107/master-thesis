import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedRegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        mask = mask[:, None, :, :].expand_as(pred)
        denom = max(mask.sum(), 1)
        loss = ((pred - gt)**2) * mask

        loss = loss.sum()/denom
        return loss

class BalancedRegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask = None):
        assert pred.size() == gt.size()

        if mask != None:
            mask = torch.flatten(mask, start_dim=2)
        pred = torch.flatten(pred, start_dim=2)
        gt = torch.flatten(gt, start_dim=2)

        joint_mask = gt >= 0.5
        
        vis_weight = joint_mask.max(2, keepdim=True)[0] * 1.0

        eu_loss = ((pred - gt)**2) * vis_weight
        joint_loss = eu_loss[joint_mask].sum()
        # body_loss = eu_loss[body_mask].sum()
        bg_loss = eu_loss[torch.logical_not(joint_mask)].sum()
        loss = joint_loss * 10 + bg_loss
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

class AnchorLoss(nn.Module):
    def __init__(self, gamma = 0.5):
        super().__init__() 
        self.gamma = gamma
        self.bce = F.binary_cross_entropy
        self.slack = 0.05
    
    def forward(self, pred, gt, mask):
        input = torch.sigmoid(pred)

        target_mask = (gt > 0.01).type('torch.FloatTensor').cuda()
        neg_mask = (1 - target_mask)

        max_mask = (gt > 0.5).type('torch.FloatTensor').cuda()
        max_pos = input * max_mask
        max_pos = max_pos.max(2)[0].max(2)[0]

        max_target = (max_mask * gt).max(2)[0].max(2)[0]
        pos = torch.ones(gt.size(0), gt.size(1)).cuda() * (1 + self.slack)
        pos[max_target > 0] = max_pos[max_target > 0]  # pos == 1 when part annotation does not exist
        pos = (pos.view(input.size(0), input.size(1), 1, 1) - self.slack).clamp(min=0).detach()

        neg_loss = -(input + (1 - pos)).pow(self.gamma) * torch.log((1 - input).clamp(min=1e-20))
        neg_loss = neg_mask * neg_loss

        pos_loss = -torch.log(input.clamp(min=1e-20))
        bg_loss = -torch.log((1-input).clamp(min=1e-20))

        bg_loss = target_mask * bg_loss

        loss = (gt * pos_loss + (1 - gt) * bg_loss) + neg_loss

        return loss.mean()



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
