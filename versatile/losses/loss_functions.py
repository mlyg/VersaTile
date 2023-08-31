import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torchvf.losses.dice import DiceLoss
from torchvf.numerics.differentiation.finite_differences import finite_differences


class StainConsistencyLossMSE(nn.Module):
    """ MSE loss for stain consistency learning"""
    def __init__(self, sigmoid=False):
        super(StainConsistencyLossMSE, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, p: torch.tensor, q: torch.tensor):
        if self.sigmoid:
            p = F.logsigmoid(p).exp()
            q = F.logsigmoid(q).exp()
        return nn.MSELoss()(p, q)


class StainConsistencyLossMAE(nn.Module):
    """ MAE loss for stain consistency learning"""
    def __init__(self, sigmoid=False):
        super(StainConsistencyLossMAE, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, p: torch.tensor, q: torch.tensor):
        if self.sigmoid:
            p = F.logsigmoid(p).exp()
            q = F.logsigmoid(q).exp()
        return nn.L1Loss()(p, q)


class StainConsistencyLossMSEx(nn.Module):
    """ MSE loss excluding background for stain consistency learning"""
    def __init__(self, sigmoid=True):
        super(StainConsistencyLossMSEx, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, p: torch.tensor, q: torch.tensor, sem_label):
        if self.sigmoid:
            p = F.logsigmoid(p).exp()
            q = F.logsigmoid(q).exp()
        loss = p - q
        loss = sem_label * (loss * loss)
        loss = loss.sum() / (sem_label.sum() + 1.0e-8)
        return loss


class StainConsistencyLossMAEx(nn.Module):
    """ MAE loss excluding background for stain consistency learning"""
    def __init__(self, sigmoid=True):
        super(StainConsistencyLossMAEx, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, p: torch.tensor, q: torch.tensor, sem_label):
        if self.sigmoid:
            p = F.logsigmoid(p).exp()
            q = F.logsigmoid(q).exp()
        loss = torch.abs(p - q)
        loss = sem_label * loss
        loss = loss.sum() / (sem_label.sum() + 1.0e-8)
        return loss


class DicePlusPlus(nn.Module):
    """ Dice++ loss 
    Overcomes calibration issue associated with conventional Dice loss
    Paper: https://arxiv.org/abs/2111.00528
    """

    def __init__(self, from_logits=True, smooth = 0.0, eps = 1e-7, alpha=0.5, beta=0.5, gamma=2):
        super(DicePlusPlus, self).__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.gamma = gamma

    def forward(self, y_pred, y_true):

        if self.from_logits:
            y_pred = F.logsigmoid(y_pred).exp()
        
        bs  = y_true.size(0)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(y_pred * y_true, dim = (0, 2))  
        fp = torch.sum((y_pred * (1.0 - y_true))** self.gamma, dim = (0, 2)) 
        fn = torch.sum(((1 - y_pred) * y_true)** self.gamma, dim = (0, 2)) 

        dice_score = (2 * tp + self.smooth) / (2*tp + fp + fn + self.smooth).clamp_min(self.eps)

        loss = 1.0 - dice_score

        mask = y_true.sum((0, 2)) > 0
        loss = loss * mask.to(loss.dtype)

        return loss.mean()


class TopKLoss(nn.Module):
    """
    Computes cross entropy loss restricted to worst pixels
    Implementation from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/compound_losses.py
    """
    def __init__(self, lb=0.1, ub=0.):
        super(TopKLoss, self).__init__()
        self.lb = lb
        self.ub = ub

    def forward(self, inputs, targets):
        res = nn.BCEWithLogitsLoss(reduction='none')(inputs,targets)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.lb ), sorted=True)
        ub_num = int(num_voxels * self.ub)
        return res[ub_num:].mean()


class FocalLoss(nn.Module):
    """ Focal loss PyTorch implementation
    Implementation: https://github.com/CoinCheung/pytorch-loss
    Paper: https://arxiv.org/abs/1708.02002

    Focal loss is useful for class imbalanced datasets.
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class DiceBCELoss(nn.Module):
    """ Compound loss using Dice and cross entropy loss"""
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets):

        dice_loss = DiceLoss(from_logits = True)(inputs, targets)
        ce_loss = nn.BCEWithLogitsLoss()(inputs,targets)
       
        return dice_loss + ce_loss

class DiceFocalLoss(nn.Module):
    """ Compound loss using Dice and Focal loss"""
    def __init__(self):
        super(DiceFocalLoss, self).__init__()
    
    def forward(self, inputs, targets):

        dice_loss = DiceLoss(from_logits = True)(inputs, targets)
        focal_loss = FocalLoss()(inputs, targets)

        return dice_loss + focal_loss


class DiceTopKLoss(nn.Module):
    """ Compound loss using Dice and TopK loss"""
    def __init__(self):
        super(DiceTopKLoss, self).__init__()
    
    def forward(self, inputs, targets):

        dice_loss = DiceLoss(from_logits = True)(inputs, targets)
        topk_loss = TopKLoss()(inputs, targets)

        return dice_loss + topk_loss


class MSDExLoss(nn.Module):
    """ MSE loss excluding background for distance transform map"""
    def __init__(self):
        super(MSDExLoss, self).__init__()
    
    def forward(self, inputs, targets, sem_label):
        sem_label = torch.cat([sem_label, sem_label], axis=1)
        loss = targets - inputs
        loss = sem_label * (loss * loss)
        loss = loss.sum() / (sem_label.sum() + 1.0e-8)
        return loss


class MADExLoss(nn.Module):
    """ MAE loss excluding background for distance transform map"""
    def __init__(self):
        super(MADExLoss, self).__init__()
    
    def forward(self, inputs, targets, sem_label):
        sem_label = torch.cat([sem_label, sem_label], axis=1)
        loss = torch.abs(targets - inputs)
        loss = sem_label * loss
        loss = loss.sum() / (sem_label.sum() + 1.0e-8)
        return loss


class MAGELoss(nn.Module):
    """ MAE loss for gradient transform map"""
    def __init__(self, device, kernel_size=3):
        super(MAGELoss, self).__init__()

        self.device = device
        self.kernel_size = kernel_size
    
    def forward(self, inputs, targets):
        inputs = finite_differences(inputs, self.kernel_size, device=self.device)
        targets = finite_differences(targets, self.kernel_size, device=self.device)

        loss = torch.abs(targets - inputs)
        return loss.mean()


class MSGELoss(nn.Module):
    """ MSE loss for gradient transform map"""
    def __init__(self, device, kernel_size=3):
        super(MSGELoss, self).__init__()

        self.device = device
        self.kernel_size = kernel_size
    
    def forward(self, inputs, targets):
        inputs = finite_differences(inputs, self.kernel_size, device=self.device)
        targets = finite_differences(targets, self.kernel_size, device=self.device)

        loss = targets - inputs
        loss = (loss * loss)
        return loss.mean()


class MAGExLoss(nn.Module):
    """ MAE loss excluding background for gradient transform map"""
    def __init__(self, device, kernel_size=3):
        super(MAGExLoss, self).__init__()

        self.device = device
        self.kernel_size = kernel_size
    
    def forward(self, inputs, targets, sem_label):
        inputs = finite_differences(inputs, self.kernel_size, device=self.device)
        targets = finite_differences(targets, self.kernel_size, device=self.device)

        sem_label = torch.cat([sem_label, sem_label], axis=1)
        loss = torch.abs(targets - inputs)
        loss = sem_label * loss
        loss = loss.sum() / (sem_label.sum() + 1.0e-8)
        return loss


class MSGExLoss(nn.Module):
    """ MSE loss excluding background for gradient transform map"""
    def __init__(self, device, kernel_size=3):
        super(MSGExLoss, self).__init__()

        self.device = device
        self.kernel_size = kernel_size
    
    def forward(self, inputs, targets, sem_label):
        inputs = finite_differences(inputs, self.kernel_size, device=self.device)
        targets = finite_differences(targets, self.kernel_size, device=self.device)

        sem_label = torch.cat([sem_label, sem_label], axis=1)
        loss = targets - inputs
        loss = sem_label * (loss * loss)
        loss = loss.sum() / (sem_label.sum() + 1.0e-8)
        return loss


