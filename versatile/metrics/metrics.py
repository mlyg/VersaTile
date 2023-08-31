import torch


def f1(output, target, smooth=1e-5):
    """ F1 or Dice score. 
    Code is from: https://github.com/ryanirl/torchvf/blob/main/torchvf/metrics/f1.py
    """
    B, C, H, W = target.shape

    output = output.reshape(B, C, -1)
    target = target.reshape(B, C, -1)

    tp = torch.sum(output * target, dim = 2)
    fp = torch.sum(output, dim = 2) - tp
    fn = torch.sum(target, dim = 2) - tp

    # IoU Score
    score = (2 * tp) / ((2 * tp) + fp + fn + smooth)

    return score 