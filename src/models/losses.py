import torch
import torch.nn as nn


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

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        mask = mask[:, None, :, :].expand_as(pred)
        pos_ind = mask >= 0.5
        pos_num = max(pos_ind.sum(),1)
        neg_ind = torch.logical_not(pos_ind)
        neg_num = max(neg_ind.sum(),1)

        eu_loss = ((pred - gt)**2)
        loss = eu_loss[pos_ind].sum() / pos_num + eu_loss[neg_ind].sum() / neg_num
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
    def __init__(self, gamma = 2.0):
        super().__init__() 
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        # mask = mask[:, None, :, :].expand_as(pred)
        
        pos_cls_mask = (gt >= 0.01) * 1.0
        neg_cls_mask = (gt < 0.01) * 1.0

        q_star =  (gt >= 0.5) * gt
        
        scale = neg_cls_mask * (1 + pred-q_star).pow(self.gamma) + pos_cls_mask
        eu_loss = ((pred - gt)**2)
        loss = scale * eu_loss
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
