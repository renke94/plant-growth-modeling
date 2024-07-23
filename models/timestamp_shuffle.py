import torch
from torch import Tensor
from torch.distributions import Binomial


class TimestampShuffle:
    def __init__(self, n: int, p: float, clip: tuple = None):
        self.distribution = Binomial(torch.tensor([n]), torch.tensor([p]))
        self.offset = n // 2
        self.clip = clip

    def __call__(self, t: Tensor) -> Tensor:
        sample = self.distribution.sample(sample_shape=t.shape).to(t.device) - self.offset
        t = t + sample.squeeze(-1).int()
        if self.clip is not None:
            t = t.clip(*self.clip)
        return t