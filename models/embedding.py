import numpy as np
import torch
from torch import nn, Tensor, IntTensor
from scipy.ndimage import gaussian_filter1d

from .common_layers import Swish


class LearnableEmbedding(nn.Module):
    def __init__(self, num_embeddings, *embedding_dim):
        super(LearnableEmbedding, self).__init__()
        self.params = nn.Parameter(torch.randn(num_embeddings, *embedding_dim))

    def forward(self, input: IntTensor) -> Tensor:
        return self.params[input]


class ContinuousEmbedding(nn.Module):
    def __init__(self, n: int, *shape, transform=None):
        super(ContinuousEmbedding, self).__init__()
        assert n >= 1, "You need at least one parameter to define the embedding"
        self.a = nn.Parameter(torch.randn(n, *shape))
        self.b = nn.Parameter(torch.randn(n, *shape))
        self.s = nn.Parameter(torch.randn(n, *shape))
        self.repeat_dims = (1,) * (len(shape) + 1)
        self.actf = Swish()
        self.transform = transform
        self.flatten = lambda t: t.view(-1, *shape)

    def forward(self, t: Tensor) -> Tensor:
        dim = t.ndim
        shape = t.shape
        t = t.view(*shape, *self.repeat_dims)
        a = self.a.repeat(*t.shape)
        emb = self.actf(a * t + self.b) * self.s
        emb = emb.sum(dim=dim)
        if self.transform is not None:
            emb = self.flatten(emb)
            emb = self.transform(emb)
            emb = emb.view(*shape, *emb.shape[1:])
        return emb


class WeightedEmbedding(nn.Module):
    def __init__(self, kernel_size: int, sigma: float, num_embeddings: int, *shape):
        super(WeightedEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.kernel_size = kernel_size
        kernel_index = torch.arange(0, kernel_size, dtype=torch.int)
        self.register_buffer('kernel_index', kernel_index)
        self.params = nn.Parameter(torch.randn(num_embeddings + kernel_size, *shape))

        weights = np.zeros(kernel_size)
        weights[kernel_size // 2] = 1
        weights = torch.from_numpy(gaussian_filter1d(weights, sigma)).float()
        self.register_buffer('weights', weights)

        self._p = "abcdefghijklmnop"[:len(shape)]

    def weighted_sum(self, t: Tensor) -> Tensor:
        p = self._p
        return torch.einsum(f'K,...K{p}->...{p}', self.weights, t)

    def forward(self, t: torch.IntTensor) -> Tensor:
        r = [1] * t.ndim + [self.kernel_size]
        t = t.unsqueeze(-1)
        t = t.repeat(*r) + self.kernel_index
        p = self.params[t]
        p = self.weighted_sum(p)
        return p


