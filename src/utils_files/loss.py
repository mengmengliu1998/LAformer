from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, List, Tuple, NamedTuple, Any


class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class GaussianNLLLoss(nn.Module):
    """https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    """
    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = 0.5*(torch.log(scale**2) + torch.abs(target - loc)**2 / scale**2)
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

def get_angle_diff(gt_traj, pred_traj, past_traj):
    """
    :param gt_traj: [B, T, 2]
    :param pred_traj: [B, T, 2]
    :param past_traj: [F, B, T, 2]
    :return: angle_diff: [B, T]
    """
    top = 5
    gt_traj_angle = gt_traj[:,:,:] - past_traj[0, :, -1, :].unsqueeze(1) # [B, T, 2]
    pred_traj_angle = pred_traj[:,:,:] - past_traj[0, :, -1, :].unsqueeze(1) # [B, T, 2]
    angle_label = torch.atan2(gt_traj_angle[:, :, 1], gt_traj_angle[:, :, 0]).to(torch.float32) #[B, T]
    angle_pred = torch.atan2(pred_traj_angle[:, :, 1], pred_traj_angle[:, :, 0]).to(torch.float32) #[B, T]
    angle_diff = angle_label - angle_pred
    angle_loss = -1 * torch.cos(angle_diff).mean(dim=-1)
    return angle_loss