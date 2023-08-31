import torch
import torch.nn.functional as F

def mean_kernel(kernel_size):
    """
    Code is from https://github.com/ryanirl/torchvf/tree/main
    """
    assert kernel_size % 2, "`kernel_size` must be divisible by 2."

    ones_kernel = torch.ones((1, kernel_size, kernel_size))
    mean_kernel = ones_kernel / ones_kernel.numel()

    return mean_kernel  


def gaussian_kernel(kernel_size, sigma = 0.5):
    """
    Code is from https://github.com/ryanirl/torchvf/tree/main
    """
    assert kernel_size % 2, "`kernel_size` must be divisible by 2."

    t    = torch.linspace(-1, 1, kernel_size)
    x, y = torch.meshgrid(t, t, indexing = "ij")
    dst  = torch.sqrt(x * x + y * y)

    gaussian_kernel = torch.exp(-(dst ** 2 / (2.0 * sigma ** 2)))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel


def finite_diff_kernel(kernel_size):
    """
    Code is from https://github.com/ryanirl/torchvf/tree/main
    """
    assert kernel_size % 2, "`kernel_size` must be divisible by 2."

    a = torch.ones((kernel_size, kernel_size // 2))
    b = torch.zeros((kernel_size, 1))
    c = -a.clone()

    finite_x = torch.cat([c, b, a], axis = 1)
    finite_y = finite_x.T

    return finite_x, finite_y


def finite_gaussian_kernel(kernel_size, sigma = 0.5):
    """
    Code is from https://github.com/ryanirl/torchvf/tree/main
    """
    assert kernel_size % 2, "`kernel_size` must be divisible by 2."

    finite_x, finite_y = finite_diff_kernel(kernel_size)
    g_kernel = gaussian_kernel(kernel_size, sigma)

    finite_x = finite_x * g_kernel
    finite_y = finite_y * g_kernel

    return finite_x, finite_y


def get_sobel_kernel(size, device='cpu'):
        """Get sobel kernel with a given size.
        Code is from  https://github.com/vqdang/hover_net
        """
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v


def sobel_finite_difference(x, kernel_size, device = "cpu"):
    kernel_h, kernel_v = get_sobel_kernel(kernel_size, device)

    sobel = torch.stack([kernel_v, kernel_h], dim = 0).to(device)

    out = F.conv2d(x, sobel[:, None], padding = kernel_size // 2)

    return out

