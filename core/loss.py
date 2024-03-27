import torch
import torch.nn as nn
EPS = 1.1920929e-07


class HeatMapJointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(HeatMapJointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
                        
            if self.use_target_weight:
                tw = target_weight[:, idx]
                    
                loss_j= self.criterion(
                    heatmap_pred.mul(tw),
                    heatmap_gt.mul(tw)
                )
            else:
                loss_j= self.criterion(heatmap_pred, heatmap_gt)

            loss += loss_j
            
        return loss.mean() / num_joints


class J3dMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(J3dMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):        
        num_joints = output.size(1)

        cnt = 0
        loss = 0
        for idx in range(num_joints):
            j3d_pred = output[:, idx]
            j3d_gt = target[:, idx]
            
            if self.use_target_weight:
                tw = target_weight[:, idx]
                
                loss += self.criterion(
                    j3d_pred.mul(tw),
                    j3d_gt.mul(tw)
                )
            else:
                loss += self.criterion(j3d_pred, j3d_gt)

        return loss.mean() / num_joints


class SegmentationLoss(nn.Module):
    def __init__(self) -> None:
        super(SegmentationLoss, self).__init__()

        self.loss = nn.BCELoss()

    def forward(self, output, target, weight=None):
        if weight is None:
            return self.loss(output, target)
        else:
            weight = weight.view(-1, 1, 1, 1)
            return self.loss(output * weight, target * weight)
  