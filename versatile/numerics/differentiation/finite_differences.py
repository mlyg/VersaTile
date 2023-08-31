import torch
import torch.nn.functional as F

from numerics.differentiation.kernels import *


def gaussian_smoothed_finite_differences(x, kernel_size, device = "cpu"):
    """
    Code is from https://github.com/ryanirl/torchvf/tree/main
    """
    gaussian_x, gaussian_y = finite_gaussian_kernel(kernel_size)

    gaussian = torch.stack([gaussian_x, gaussian_y], dim = 0).to(device)

    out = F.conv2d(x, gaussian[:, None], padding = kernel_size // 2)

    return out


def finite_differences(x, kernel_size, device = "cpu"):
    """
    Code is from https://github.com/ryanirl/torchvf/tree/main
    """
    finite_x, finite_y = finite_diff_kernel(kernel_size)

    finite = torch.stack([finite_x, finite_y], dim = 0).to(device)

    out = F.conv2d(x, finite[:, None], padding = kernel_size // 2)

    return out


