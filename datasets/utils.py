from pathlib import Path

import torch
from PIL import Image
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import Tensor
from torchvision.transforms.functional import to_tensor


def load_image(path: str | Path) -> torch.Tensor:
    """
    Loads an image from the specified path and converts it to a tesnor.

    Args:
        path (str | Path): Path specifying the path to the image

    Returns:
        torch.Tensor: Tensor containing the image data
    """
    return to_tensor(Image.open(path))

def load_latent(path: str | Path) -> torch.Tensor:
    """
    Loads a latent variable from a file and returns a sample from the resulting diagonal Gaussian distribution.

    Args:
        path (str | Path): Path specifying the path to the latent variable

    Returns:
        torch.Tensor: Sample from the Gaussian distribution, defined by the loaded latent values.
    """
    latent = torch.load(path).unsqueeze(0)
    latent = DiagonalGaussianDistribution(latent).sample() * 0.18215
    return latent


def rgbvi(image: Tensor) -> Tensor:
    """
    Calculated the RGBVI value of a given RGB image.
    The values are calculated pixel-wise as follows:
    RGBVI = (G^2 - R * B) / (G^2 + R * B)

    Args:
        image: Tensor of shape [..., 3, H, W] that can have any shape as long the RGB channels are the dimension at
        index -3
    Returns:
        Tensor of shape [..., 1, H, W] containing the RGBVI values of each pixel.
    """
    assert image.size(-3) == 3, "image must have 3 channels"
    r, g, b = image.chunk(3, dim=-3)
    g2 = g**2
    rb = r*b
    return (g2 - rb) / (g2 + rb)

def batch_dim_collate_factory(dim: int):
    """
    Generates a collate function for torch DataLoaders, that stacks the samples along the specified dimension.

    Args:
        dim (int): determines the batch dimension
    Returns:
        Callable that takes a batch and stacks its samples along the given dimension
    """
    def collate_fn(batch):
        batch = [torch.stack(l, dim=dim) for l in zip(*batch)]
        return batch

    return collate_fn