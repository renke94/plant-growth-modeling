import torch
import torch.nn.functional as F
from torch import nn, Tensor


class GlobalAveragePooling(nn.Module):
    """
    Implementation of global average pooling for PyTorch.
    """
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, inputs):
        return inputs.mean(dim=(-2, -1), keepdim=True)


class GlobalMaxPooling(nn.Module):
    """
    Implementation of global max pooling for PyTorch.
    """
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()

    def forward(self, inputs):
        return inputs.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values


class Lambda(nn.Module):
    """
    Wrapper module that can easily be used within torch.nn.Sequential modules.
    It performs the operations defined by the function passed in the constructor.

    Args:
         function (Callable): A function that takes a tensor as input and returns a tensor.
    """
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, inputs):
        return self.function(inputs)


class DepthWiseConv2d(nn.Module):
    """
    Depth-wise version of PyTorch's Conv2d module where each input channel is convolved with its own set of filters.
    Usage is almost the same as that of traditional Conv layers.

    Args:
        c_in (int): Number of input channels
        c_out (int): Number of output channels
        kernel_size (int): size of the convolution kernel
        padding (int): padding for the inputs
        stride (int): stride in which the convolution is applied
        bias (bool): whether to use bias or not
    """
    def __init__(
            self,
            c_in: int,
            c_out: int,
            kernel_size: int,
            padding: int = 0,
            stride: int = 1,
            bias: bool = True,
    ):
        super(DepthWiseConv2d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, padding=padding, groups=c_in, stride=stride, bias=bias),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Swish(nn.Module):
    """
    Swish activation function from https://arxiv.org/abs/1710.05941v2
    """
    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class ResidualBlock(nn.Module):
    """
    Residual block consisting of two convolution, group normalization, and swish activations, surrounded by a shortcut
    connection.

    Args:
        c_in(int): Number of input channels
        c_out(int): Number of output channels
    """
    def __init__(self, c_in: int, c_out: int, *args, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, c_in),
            Swish(),
            nn.Conv2d(c_in, c_out, 3, stride=1, padding=1),
            nn.GroupNorm(32, c_out),
            Swish(),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1),
        )
        if c_in != c_out:
            self.shortcut = nn.Conv2d(c_in, c_out, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.shortcut(x)


class UpSample(nn.Module):
    """
    Up sampling layer, that scales up a spatial tensor with a scale factor of 2.0.
    The upsampling is performed by a nearest-neighbor interpolation followed by a convolution, which retains the number
    of channels.

    Args:
        channels(int): Number of channels needed for the convolution.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownSample(nn.Module):
    """
    Down sampling layer, that down samples a spatial tensor with a scale factor of 2.0 using strided convolution.
    The number of channels is retained during this operation.

    Args:
        channels(int): Number of channels needed for the convolution.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)
