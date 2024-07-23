import torch.nn.functional as F
from einops import rearrange, einsum
from torch import nn, Tensor

from .common_layers import DepthWiseConv2d


class QKVAttention2d(nn.Module):
    """
    QKV attention for 2d inputs base class that provides the linear input projections for query, key, and value as well
    as the linear output projection.

    Must be inherited by subclasses that implement the forward function.

    Args:
        d_q (int): Dimensionality of the query input channels
        d_k (Optional[int]): Dimensionality of the key input channels. If None: d_k = d_q
        d_v (Optional[int]): Dimensionality of the value input channels which also determine the output dimensionality. If None: d_v = d_q
        nhead (int): Number of heads
        dim_head (int): Dimensionality of each head
        dropout (float): Dropout rate during training
    """
    def __init__(self,
                 d_q: int,
                 d_k: int = None,
                 d_v: int = None,
                 nhead: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.0):
        super(QKVAttention2d, self).__init__()
        if d_k is None:
            d_k = d_q
        if d_v is None:
            d_v = d_q
        self.nhead = nhead
        self.scale = dim_head ** -0.5
        self.inner_dim = nhead * dim_head
        self.dropout = nn.Dropout(dropout)
        self.q = nn.Conv2d(d_q, self.inner_dim, kernel_size=1, bias=False)
        # self.k = nn.Conv2d(d_k, self.inner_dim, kernel_size=1, bias=False)
        # self.v = nn.Conv2d(d_v, self.inner_dim, kernel_size=1, bias=False)
        self.k = DepthWiseConv2d(d_k, self.inner_dim, kernel_size=3, bias=False)
        self.v = DepthWiseConv2d(d_v, self.inner_dim, kernel_size=3, bias=False)
        self.to_out = nn.Conv2d(self.inner_dim, d_q, kernel_size=1)
        self.act = nn.GELU()

        self.rearrange = lambda t: rearrange(t, 'B (n C) H W -> (B n) (H W) C', n=nhead)
        self.restructure = lambda t, H: rearrange(t, '(B n) (H W) C -> B (n C) H W', n=nhead, H=H)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        raise NotImplementedError()


class EfficientAttention2d(QKVAttention2d):
    """
    Efficient attention module for spatial data, implemented on the basis of https://arxiv.org/abs/1812.01243
    Implementation is extended with multi-head attention.

    In principle this module calculates the traditional QKV scaled dot product attention with linear complexities.
    Since matrix multiplication is associative, (Q K^T) V = Q (K^T V) applies.
    This implementation has the trade-off, that softmax scaling is only approximated.

    Args:
        d_q (int): Dimensionality of the query input channels
        d_k (Optional[int]): Dimensionality of the key input channels. If None: d_k = d_q
        d_v (Optional[int]): Dimensionality of the value input channels which also determine the output dimensionality. If None: d_v = d_q
        nhead (int): Number of heads
        dim_head (int): Dimensionality of each head
        dropout (float): Dropout rate during training

    Examples:
        >>> import torch
        >>> from models.attention import EfficientAttention2d
        >>> q = torch.randn(8, 128, 32, 32)
        >>> k = v = torch.randn(8, 128, 64, 64)
        >>> attn = EfficientAttention2d(128)
        >>> out = attn(q, k, v)
        >>> print(out.shape)
        >>> torch.Size([8, 128, 32, 32])
    """
    def __init__(self,
                 d_q: int,
                 d_k: int = None,
                 d_v: int = None,
                 nhead: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.0):
        super(EfficientAttention2d, self).__init__(d_q, d_k, d_v, nhead, dim_head, dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Applies multi-head scaled dot product attention on the spatial dimensions of the inputs.
        All inputs must have type: torch.Tensor
        Args:
            q: [B C H' W']
            k: [B C H W]
            v: [B C H W]

        Returns: [B C H' W'] Tensor of the same shape as q
        """
        H = q.shape[-2]
        q = self.rearrange(self.q(q)).softmax(-1) * self.scale  # [B M C] softmax channels
        k = self.rearrange(self.k(k)).softmax(-2)               # [B N C] softmax sequence
        v = self.rearrange(self.v(v))                           # [B N C]

        q = self.dropout(q)

        weights = einsum(k, v, 'B N C, B N D -> B C D')         # K^T V
        out = einsum(q, weights, 'B M C, B C D-> B M D')        # Q (K^T V)
        out = self.restructure(out, H=H)
        out = self.act(out)
        out = self.to_out(out)
        return out


class SpatialAttention2d(QKVAttention2d):
    """
    Scaled dot-product multi-head attention module for 2d inputs, implemented on the basis of
    https://arxiv.org/abs/1706.03762

    Args:
        d_q (int): Dimensionality of the query input channels
        d_k (Optional[int]): Dimensionality of the key input channels. If None: d_k = d_q
        d_v (Optional[int]): Dimensionality of the value input channels which also determine the output dimensionality. If None: d_v = d_q
        nhead (int): Number of heads
        dim_head (int): Dimensionality of each head
        dropout (float): Dropout rate during training

    Examples:
        >>> import torch
        >>> from models.attention import SpatialAttention2d
        >>> q = torch.randn(8, 128, 32, 32)
        >>> k = v = torch.randn(8, 128, 64, 64)
        >>> attn = SpatialAttention2d(128)
        >>> out = attn(q, k, v)
        >>> print(out.shape)
        >>> torch.Size([8, 128, 32, 32])
    """
    def __init__(self,
                 d_q: int,
                 d_k: int = None,
                 d_v: int = None,
                 nhead: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.0):
        super(SpatialAttention2d, self).__init__(d_q, d_k, d_v, nhead, dim_head, dropout)
        self.dropout_p = dropout

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Applies multi-head scaled dot product attention on the spatial dimensions of the inputs.
        All inputs must have type: torch.Tensor
        Args:
            q: [B C H' W']
            k: [B C H W]
            v: [B C H W]

        Returns: [B C H' W'] Tensor of the same shape as q
        """
        H = q.shape[-2]
        q = self.rearrange(self.q(q))   # [B M C]
        k = self.rearrange(self.k(k))   # [B N C]
        v = self.rearrange(self.v(v))   # [B N C]

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self.restructure(out, H=H)
        out = self.act(out)
        out = self.to_out(out)
        return out


class ChannelAttention2d(QKVAttention2d):
    """
    Scaled dot-product multi-head attention module for 2d inputs, with the attention weights layed over the channel
    dimention.

    Args:
        d_q (int): Dimensionality of the query input channels
        d_k (Optional[int]): Dimensionality of the key input channels. If None: d_k = d_q
        d_v (Optional[int]): Dimensionality of the value input channels which also determine the output dimensionality. If None: d_v = d_q
        nhead (int): Number of heads
        dim_head (int): Dimensionality of each head
        dropout (float): Dropout rate during training

    Examples:
        >>> import torch
        >>> from models.attention import ChannelAttention2d
        >>> q = torch.randn(B, 64, 32, 32)
        >>> k = torch.randn(B, 128, 32, 32)
        >>> v = torch.randn(B, 32, 64, 64)
        >>> attn = ChannelAttention2d(64, 128, 32)
        >>> out = attn(q, k, v)
        >>> print(out.shape)
        >>> torch.Size([8, 32, 64, 64])
    """
    def __init__(self,
                 d_q: int,
                 d_k: int = None,
                 d_v: int = None,
                 nhead: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.0):
        super(ChannelAttention2d, self).__init__(d_q, d_k, d_v, nhead, dim_head, dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Applies multi-head scaled dot product attention on the channel dimension of the inputs.
        All inputs must have type: torch.Tensor
        Args:
            q: [B C H W]
            k: [B C H W]
            v: [B C H' W']

        Returns: [B C H' W'] Tensor of the same shape as v
        """
        H = v.shape[-2]
        q = self.rearrange(self.q(q))  # [B M C]
        k = self.rearrange(self.k(k))  # [B N C]
        v = self.rearrange(self.v(v))  # [B N C]

        q = self.dropout(q)

        weights = einsum(q, k, 'B N C, B N D -> B C D') * q.shape[-2] ** -0.5
        weights = weights.softmax(-1)
        out = einsum(weights, v, 'B C D, B N C -> B N D')
        out = self.restructure(out, H=H)
        out = self.act(out)
        out = self.to_out(out)
        return out


