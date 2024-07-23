from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .attention import EfficientAttention2d
from .common_layers import Swish, DownSample, UpSample, ResidualBlock
from .embedding import ContinuousEmbedding, LearnableEmbedding
from .lgm import LatentGrowthModel, LatentGrowthVAE


def make_unet_features(down_features: list[int], up_features: list[int]):
    """
    Helper function for calculating UNet channel inputs and outputs.
    Args:
        down_features (List[int]): list of channel transitions for the UNet downsampling
        up_features (List[int]): list of channel transitions for the UNet upsampling

    Returns:
        down_features (List[Tuple[int, int]]): list of tuples of channel transitions for the UNet downsampling
            in the form [(c_in, c_out), ...]
        down_features (List[Tuple[int, int]]): list of tuples of channel transitions for the UNet upsampling
            in the form [(c_in, c_out), ...], where the number of shortcut channels is added at each level.

    Examples:
        >>> from models.lg_unet import make_unet_features
        >>> print(make_unet_features([2, 4, 8], [8, 4, 2]))
        ([(2, 4), (4, 8)], [(16, 4), (8, 2)])
    """
    down_c_in = np.array(down_features[:-1])
    down_c_out = np.array(down_features[1:])
    up_c_in = np.array(up_features[:-1])
    up_c_out = np.array(up_features[1:])
    up_c_in = up_c_in + down_c_out[::-1]
    f_down = list(zip(down_c_in, down_c_out))
    f_up = list(zip(up_c_in, up_c_out))
    return f_down, f_up

class LGUnetBase(LatentGrowthModel):
    """
    Abtract UNet class with basic functionality for down- and upsampling.
    """
    def __init__(self, dropout=0.1):
        super(LGUnetBase, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def downsample(self, z: Tensor, conditioning: Optional[Tensor] = None) -> tuple[Tensor, list[Tensor]]:
        """
        Downsamples an input z and returns output of the last layer and a list of intermediate feature maps.
        Args:
            z (torch.Tensor):
            conditioning (Optional[torch.Tensor]):

        Returns:
            z (torch.Tensor):
            fmaps (List[torch.Tensor])
        """
        fmaps = []
        for i in range(0, len(self.down_blocks), 2):
            block, down = self.down_blocks[i:i+2]
            z = block(z, conditioning)
            z = self.dropout(z)
            z = down(z)
            fmaps.append(z)
        return z, fmaps[::-1]

    def upsample(self, z: Tensor, fmaps: list[Tensor], conditioning: Optional[Tensor] = None) -> Tensor:
        """
        Upsamples an input z and uses the intermediate feature maps from the downsampling as additional input features.
        Args:
            z (torch.Tensor):
            fmaps (List[torch.Tensor]):
            conditioning (Optional[torch.Tensor]): Conditioning inputs for cross attention

        Returns (torch.Tensor): upsampled version of z
        """
        for i in range(0, len(self.up_blocks), 2):
            block, up = self.up_blocks[i:i+2]
            z = block(torch.cat([z, fmaps[i // 2]], dim=-3), conditioning)
            z = up(z)
            z = self.dropout(z)
        return z


class LGUnetBlock(ResidualBlock):
    """
    Unet block whose architecture is similar to that of a Transformer block:
    1. residual self-attention block
    2. residual cross-attention block (optional)
    3. residual conv block

    Args:
        c_in (int): Number of input channels
        c_out (int): Number of output channels
        c_cond (int | None): dimensionality of the conditioning variables. If None, c_cond = c_in
        nhead (int): Number of attention heads
        dim_head (int): Dimensionality of each head, If None, dim_head = c_in // nhead
        dropout (float): Dropout rate
    """
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 c_cond: int = None,
                 nhead: int = 8,
                 dim_head: int = None,
                 dropout: float = 0.1,
                 ):
        super(LGUnetBlock, self).__init__(c_in, c_out)
        dim_head = c_in // nhead if dim_head is None else dim_head

        self.self_attention = EfficientAttention2d(
            d_q=c_in,
            nhead=nhead,
            dim_head=dim_head,
            dropout=dropout
        )

        self.norm_sa = nn.GroupNorm(32, c_in)

        self.cross_attention = EfficientAttention2d(
            d_q=c_in,
            d_k=c_cond,
            d_v=c_cond,
            nhead=nhead,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.norm_ca = nn.GroupNorm(32, c_in)

        self.ffwd = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=1, padding=1),
            nn.GroupNorm(32, c_out),
            Swish(),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1),
        )

        if c_in != c_out:
            self.shortcut = nn.Conv2d(c_in, c_out, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.norm_ffwd = nn.GroupNorm(32, c_out)

        self.swish = Swish()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: Tensor, condition: Tensor = None) -> Tensor:
        x = self.norm_sa(x + self.self_attention(q=x, k=x, v=x))
        if condition is not None:
            x = self.norm_ca(x + self.cross_attention(q=x, k=condition, v=condition))

        x = self.norm_ffwd(self.shortcut(x) + self.ffwd(x))
        x = self.swish(x)
        x = self.dropout(x)
        return x

class LGUnet(LGUnetBase):
    """
    Latent Growth Unet desidned to model growth in latent space.
    The Unet is provided with a sequence of 4 latents, defining the context.
    For additional information, timestamps that correspond to the input sequence must be passed.
    The output depends on the requested times,

    Args:
        input_shape (tuple[int]): Shape of the latent variables
        down_features (list[int]): List of channel dimensionalities of the down sampling layers
        up_features (list[int]): List of channel dimensionalities of the up sampling layers
        nhead (int): Number of attention heads
        dim_head (int): Dimensionality of each head, If None, dim_head = c_in // nhead
        n_positions (int): Number of possible timestamps that determines the size of the positional encoding.
        dropout (float): Dropout rate
    """
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 down_features: list[int],
                 up_features: list[int],
                 nhead: int = 8,
                 dim_head: int = None,
                 n_positions: int = 1000,
                 dropout: float = 0.1):
        super(LGUnet, self).__init__(dropout=dropout)
        self.pos_enc = nn.Parameter(torch.randn(n_positions, *input_shape))

        # input pipeline
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0] * 4, down_features[0], 7, padding=3),
            nn.GroupNorm(32, down_features[0]),
            Swish(),
        )

        # conditional unet
        c_cond = 8
        self.conditioning = nn.Parameter(torch.randn(n_positions, c_cond, 32, 32))

        down_features, up_features = make_unet_features(down_features, up_features)
        self.down_blocks = nn.ModuleList()
        for c_in, c_out in down_features:
            self.down_blocks.append(LGUnetBlock(c_in, c_out, c_cond, nhead, dim_head, dropout))
            self.down_blocks.append(DownSample(c_out))

        self.bottleneck = LGUnetBlock(c_out, c_out, c_cond, nhead, dim_head, dropout)

        self.up_blocks = nn.ModuleList()
        for c_in, c_out in up_features:
            self.up_blocks.append(LGUnetBlock(c_in, c_out, c_cond, nhead, dim_head, dropout))
            self.up_blocks.append(UpSample(c_out))

        # output pipeline
        self.conv_out = nn.Conv2d(c_out, input_shape[0], 3, padding=1)

    def generate(self, z: Tensor, fmaps: list[Tensor], t_out: Tensor) -> Tensor:
        """
        Generates a latent variable from z, the Unet features 'fmaps' and a conditioning vector 't_out'.
        Only for internal usage.
        """
        conditioning = self.conditioning[t_out.int()]
        z = self.bottleneck(z, conditioning)
        z = self.upsample(z, fmaps, conditioning)
        z = self.conv_out(z)
        return z

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
        z = z_in + self.pos_enc[t_in.int()]     # [B n 4 32 32]
        z = self.conv_in(z.flatten(1, 2))       # [B C 32 32]
        z, fmaps = self.downsample(z)           # z = [B C H W]

        outputs = []
        for i in range(t_out.shape[1]):
            outputs.append(self.generate(z, fmaps, t_out[:, i]))
        outputs = torch.stack(outputs, dim=1)   # [B m 4 32 32]
        if z_out is None:
            return outputs
        loss = F.mse_loss(outputs, z_out)
        return outputs, loss


class LGU(LatentGrowthVAE):
    """
    Wrapper class that combines the pre-trained AutoencoderKL with the Latent Growth U-Net model.
    This class provides a simple API for end-to-end inference in the pixel space, with pre-configured and pre-trained
    LGU models.

    Args:
        resolution (int): Resolution of the images. Must be eithe 128, 256, or 512.
        ckpt_path (str): Checkpoint path to the pre-trained LGU model.

    Example:
        >>> from dlbpgm.models import LGU
        >>> model = LGU(resolution=256, ckpt_path='path/to/checkpoint.pt')
        >>> x_in = torch.rand(1, 4, 3, 256, 256)    # [B, N, C, H, W]
        >>> t_in = torch.randint(0, 100, (1, 4))    # [B, N]
        >>> t_out = torch.randint(0, 100, (1, 3))   # [B, M]
        >>> print(model(x_in, t_in, t_out).shape)   # [B, M, C, H, W]
        torch.Size([1, 3, 3, 256, 256])
    """
    def __init__(self, resolution: int, ckpt_path: str = None):
        assert resolution in {128, 256, 512}, "Resolution must be either 128, 256 or 512"

        latent_dim = resolution // 8

        if resolution == 128:
            down_features = [128, 256, 512]
            up_features = [512, 256, 256]
        else:
            down_features = [128, 256, 512, 1024]
            up_features = [1024, 512, 256, 256]

        model = LGUnet(
            input_shape=(4, latent_dim, latent_dim),
            down_features=down_features,
            up_features=up_features,
            nhead=32,
            dim_head=16,
            dropout=0.1,
            n_positions=1000
        )


        if ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path))

        super().__init__(model)
        self.eval()


class LGUnetImprovedEmbedding(LGUnetBase):
    """
    Latent Growth Unet desidned to model growth in latent space.
    The Unet is provided with a sequence of 4 latents, defining the context.
    For additional information, timestamps that correspond to the input sequence must be passed.
    The output depends on the requested times,

    Args:
        input_shape (tuple[int]): Shape of the latent variables
        down_features (list[int]): List of channel dimensionalities of the down sampling layers
        up_features (list[int]): List of channel dimensionalities of the up sampling layers
        nhead (int): Number of attention heads
        dim_head (int): Dimensionality of each head, If None, dim_head = c_in // nhead
        n_positions (int): Number of possible timestamps that determines the size of the positional encoding.
        dropout (float): Dropout rate
    """
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 down_features: list[int],
                 up_features: list[int],
                 nhead: int = 8,
                 dim_head: int = None,
                 n_positions: int = 1000,
                 dropout: float = 0.1,
                 pos_enc: nn.Module = None,
                 conditioning: nn.Module = None,
                 ):
        super(LGUnetImprovedEmbedding, self).__init__(dropout=dropout)
        if pos_enc is None:
            self.pos_enc = LearnableEmbedding(n_positions, *input_shape)
        else:
            self.pos_enc = pos_enc

        # input pipeline
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0] * 4, down_features[0], 7, padding=3),
            nn.GroupNorm(32, down_features[0]),
            Swish(),
        )

        # conditional unet
        c_cond = 8
        if conditioning is None:
            self.conditioning = LearnableEmbedding(n_positions, (c_cond, 32, 32))
        else:
            self.conditioning = conditioning

        down_features, up_features = make_unet_features(down_features, up_features)
        self.down_blocks = nn.ModuleList()
        for c_in, c_out in down_features:
            self.down_blocks.append(LGUnetBlock(c_in, c_out, c_cond, nhead, dim_head, dropout))
            self.down_blocks.append(DownSample(c_out))

        self.bottleneck = LGUnetBlock(c_out, c_out, c_cond, nhead, dim_head, dropout)

        self.up_blocks = nn.ModuleList()
        for c_in, c_out in up_features:
            self.up_blocks.append(LGUnetBlock(c_in, c_out, c_cond, nhead, dim_head, dropout))
            self.up_blocks.append(UpSample(c_out))

        # output pipeline
        self.conv_out = nn.Conv2d(c_out, input_shape[0], 3, padding=1)

    def generate(self, z: Tensor, fmaps: list[Tensor], t_out: Tensor) -> Tensor:
        """
        Generates a latent variable from z, the Unet features 'fmaps' and a conditioning vector 't_out'.
        Only for internal usage.
        """
        conditioning = self.conditioning(t_out.int())
        z = self.bottleneck(z, conditioning)
        z = self.upsample(z, fmaps, conditioning)
        z = self.conv_out(z)
        return z

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
        z = z_in + self.pos_enc(t_in.int())     # [B n 4 32 32]
        z = self.conv_in(z.flatten(1, 2))       # [B C 32 32]
        z, fmaps = self.downsample(z)           # z = [B C H W]

        outputs = []
        for i in range(t_out.shape[1]):
            outputs.append(self.generate(z, fmaps, t_out[:, i]))
        outputs = torch.stack(outputs, dim=1)   # [B m 4 32 32]
        if z_out is None:
            return outputs
        loss = F.mse_loss(outputs, z_out)
        return outputs, loss