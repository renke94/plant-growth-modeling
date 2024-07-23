from math import log2
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .common_layers import Swish, DownSample, UpSample
from .lg_transformer import ConvBlock
from .lgm import LatentGrowthModel, LatentGrowthVAE


class LinearRegression(nn.Module):
    """
    Linear Regression as PyTorch module.
    The model is fitted on instantiation.

    Args:
        x (Tensor): Tensor of arbitrary shape
        y (Tensor): Tensor of arbitrary shape
    """
    def __init__(self, x: Tensor, y: Tensor, dim: int = 0):
        super(LinearRegression, self).__init__()
        while x.ndim < y.ndim:
            x = x.unsqueeze(-1)
        x_mean = x.mean(dim=dim, keepdim=True)
        y_mean = y.mean(dim=dim, keepdim=True)
        x_diff = x - x_mean
        y_diff = y - y_mean
        b = (x_diff * y_diff).sum(dim=dim, keepdim=True) #/ (x_diff**2).sum(dim=dim, keepdim=True)
        a = y_mean - b * x_mean
        self.register_buffer('b', b)
        self.register_buffer('a', a)

    def forward(self, x: Tensor):
        """
        Samples new values from the fitted linear regression function
        Args:
            x (Tensor): Tensor of arbitrary shape

        Returns: Tensor with same shape as y during instantiation
        """
        while x.ndim < self.b.ndim:
            x = x.unsqueeze(-1)
        return self.a + self.b * x


class LatentGrowthRegression(LatentGrowthModel):
    """
    Latent Growth Regression PyTorch Module implementing the API of the LatentGrowthModel.
    This model performes an individual downsampling of the latent variables before the linear regression is applied for
    the temporal modeling. After the regression, the latents are upsampled individually to correspond to the size of the
    inputs.

    Args:
        resolution (int): Resolution of the latent variables. This is either 16, 32, or 64, depending on the image
            resolution.
        nhead (int): Number of heads used in multi-head self- and cross-attention (default: 8).
        dim_head (int): dimensionality of each head (default: 32).
        dropout (float): Dropout probability during training (default: 0.1).
        n_positions (int): Number of embeddings to be initialized for positional encoding (default: 1000).
    """
    def __init__(self,
                 resolution: int,
                 nhead: int = 8,
                 dim_head: int = 32,
                 dropout: float = 0.1,
                 n_positions: int = 1000,
                 ):
        super(LatentGrowthRegression, self).__init__()
        n_downsample = log2(resolution)
        assert n_downsample.is_integer(), "resolution must be an exponent of 2"
        n_upsample = int(n_downsample) - 1

        features = 64 * 2**np.arange(n_upsample)
        down_features = list(zip(features[:-1], features[1:]))
        features = features[::-1]
        up_features = list(zip(features[:-1], features[1:]))

        self.downsample = nn.Sequential(
            nn.Conv2d(4, features[-1], kernel_size=3, stride=1, padding=1),   #
            nn.GroupNorm(32, features[-1]),
            Swish(),
        )

        for c_in, c_out in down_features:
            self.downsample.append(ConvBlock(c_in, c_out, nhead, dim_head, dropout))
            self.downsample.append(DownSample(c_out))

        self.upsample = nn.Sequential()

        for c_in, c_out in up_features:
            self.upsample.append(ConvBlock(c_in, c_out, nhead, dim_head, dropout))
            self.upsample.append(UpSample(c_out))

        self.upsample.append(nn.Conv2d(c_out, 4, kernel_size=3, stride=1, padding=1))

        self.pos_enc = nn.Parameter(torch.randn(n_positions, 4, resolution, resolution))


    def forward(self, z_in: Tensor, t_in: Tensor, t_out: Tensor, z_out: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Performs the growth modeling in the latent space based on the sequence of input images `z_in`, their
        corresponding timestamps `t_in` and the desired timestamps `t_out`. During training, the loss can also be
        calculated by providing the set of target images `z_in`.

        Args:
            z_in (Tensor): Tensor of shape [B N C H W] containing a batch of sequences of latent variables
            t_in (Tensor): Tensor of shape [B N] containing the corresponding timestamps for z_in
            t_out (Tensor): Tensor of shape [B M] containing the desired timestamps for the outputs
            z_out (Tensor | None): Optional Tensor of shape [B M C H W] containing the target outputs. This is passed
                during training, making this function return the predictions and the MSE loss between predictions and
                targets.

        Returns:
            Tensor of shape [B M C H W]
            If z_out is not None the loss is also be returned as a second return value.
        """
        B, N = z_in.shape[:2]
        M = t_out.shape[1]
        z = z_in + self.pos_enc[t_in.int()]
        z = rearrange(z, "B N C H W -> (B N) C H W")
        z = self.downsample(z)
        z = rearrange(z, "(B N) ... -> B N ...", N=N)
        regression = LinearRegression(t_in.float(), z, dim=1)
        z = regression(t_out.float())

        z = z.flatten(0, 1)
        z = self.upsample(z)
        z = rearrange(z, "(B M) C H W -> B M C H W", M=M)
        if z_out is None:
            return z
        loss = F.mse_loss(z, z_out)
        return z, loss


class LGR(LatentGrowthVAE):
    """
    Wrapper class that combines the pre-trained AutoencoderKL with the Latent Growth Regression model.
    This class provides a simple API for end-to-end inference in the pixel space, with pre-configured and pre-trained
    LGR models.

    Args:
        resolution (int): Resolution of the images. Must be eithe 128, 256, or 512.
        ckpt_path (str): Checkpoint path to the pre-trained LGR model.

    Example:
        >>> from dlbpgm.models import LGR
        >>> model = LGR(resolution=256, ckpt_path='path/to/checkpoint.pt')
        >>> x_in = torch.rand(1, 4, 3, 256, 256)    # [B, N, C, H, W]
        >>> t_in = torch.randint(0, 100, (1, 4))    # [B, N]
        >>> t_out = torch.randint(0, 100, (1, 3))   # [B, M]
        >>> print(model(x_in, t_in, t_out).shape)   # [B, M, C, H, W]
        torch.Size([1, 3, 3, 256, 256])
    """
    def __init__(self, resolution: int, ckpt_path: str = None):
        assert resolution in {128, 256, 512}, "Resolution must be either 128, 256 or 512"

        latent_dim = resolution // 8

        model = LatentGrowthRegression(
            resolution=latent_dim,
            nhead=32,
            dim_head=16,
            dropout=0.1,
            n_positions=1000
        )

        if ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path))

        super().__init__(model)
        self.eval()