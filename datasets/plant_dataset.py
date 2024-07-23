from pathlib import Path

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch.utils.data import Dataset

from .metadata import Metadata
from .utils import load_image


class PlantMetaDataset(Dataset):
    """
    PlantMetaDataset class that stores metadata about the image time series of one plant.
    Yields randomly sampled subsequences as inputs and targets in form of pandas DataFrames.

    Args:
        metadata (Metadata): The metadata about the time series of one plant
        timespan (int): The range in which to sample the subsequences
        n_inputs (int): Length of the input sequence
        n_targets (int): Length of the target sequence
    """
    def __init__(self,
                 metadata: Metadata,
                 timespan: int,
                 n_inputs: int,
                 n_targets: int):
        super(PlantMetaDataset, self).__init__()
        self.metadata = metadata.sort_values(by='time')
        self.timespan = timespan
        self.n_inputs = n_inputs
        self.n_targets = n_targets
        self.sequence_length = n_inputs + n_targets

    def __len__(self):
        n = len(self.metadata)
        if n < self.sequence_length:
            return 0
        if n <= self.timespan:
            return 1
        else:
            return n - self.timespan

    def __getitem__(self, idx):
        n = len(self.metadata)
        sample = idx + torch.randperm(min(self.timespan, n))[:self.sequence_length]
        x = sample[:self.n_inputs]
        y = sample[self.n_inputs:]
        x = self.metadata.iloc[x]
        y = self.metadata.iloc[y]
        return x, y


class StridedPlantMetaDataset(Dataset):
    """
    A dataset for evaluation purposes in which the timesteps between the images can be determined by a fixed stride.
    Yields the metadata of the respecting samples in form of a DataFrame.

    Args:
        metadata (Metadata): The metadata about the time series of one plant
        stride (int): The stride in which to sample the images
        sequence_length (int): The length of yielded sequences
    """
    def __init__(self,
                 metadata: Metadata,
                 stride: int,
                 sequence_length: int):
        self.metadata = metadata.sort_values(by='time')
        self.stride = stride
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.metadata) - self.sequence_length * self.stride)

    def __getitem__(self, idx):
        sample = idx + np.arange(self.sequence_length) * self.stride
        sample = self.metadata.iloc[sample]
        return sample


class StridedPlantDataset(StridedPlantMetaDataset):
    """
    A dataset for evaluation purposes in which the timesteps between the images can be determined by a fixed stride.
    Yields the samples in form of x_in, t_in, t_out, x_out where all variables are Tensors.
    Inputs and targets are chosen in an alternating way from the sample

    Args:
        metadata (Metadata): The metadata about the time series of one plant
        stride (int): The stride in which to sample the images
        sequence_length (int): The length of yielded sequences
        transform (callable, optional): A function/transform that is applied to the images.
    """
    def __init__(self,
                 metadata: Metadata,
                 stride: int,
                 sequence_length: int,
                 transform=None):
        super(StridedPlantDataset, self).__init__(metadata, stride, sequence_length)
        self.transform = transform

    def __getitem__(self, idx):
        sample = super(StridedPlantDataset, self).__getitem__(idx)
        x = torch.stack([load_image(p) for p in sample.path])
        t = torch.tensor(sample.time.to_numpy())
        if self.transform is not None:
            x = self.transform(x)
        x_in = x[1::2]
        x_out = x[::2]
        t_in = t[1::2]
        t_out = t[::2]
        return x_in, t_in, t_out, x_out


class PlantDataset(PlantMetaDataset):
    """
    PlantDataset class that stores metadata about the image time series of one plant.
    Yields randomly sampled subsequences as inputs and targets in form of images and their corresponding times.

    Args:
        metadata (Metadata): The metadata about the time series of one plant
        timespan (int): The range in which to sample the subsequences
        n_inputs (int): Length of the input sequence
        n_targets (int): Length of the target sequence
    """
    def __init__(self,
                 metadata: Metadata,
                 timespan: int,
                 n_inputs: int,
                 n_targets: int,
                 transform=None):
        super(PlantDataset, self).__init__(metadata, timespan, n_inputs, n_targets)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = super(PlantDataset, self).__getitem__(idx)
        t_in = torch.tensor(x.time.to_numpy())
        t_out = torch.tensor(y.time.to_numpy())
        x_in = torch.stack([load_image(p) for p in x.path])
        x_out = torch.stack([load_image(p) for p in y.path])
        if self.transform:
            # it's important to concat and split inputs and outputs
            # to ensure they are transformed the same way
            x_trans = self.transform(torch.cat([x_in, x_out]))
            x_in, x_out = x_trans.split([x_in.shape[0], x_out.shape[0]])
        return x_in, t_in, t_out, x_out

class PlantLatentDataset(PlantMetaDataset):
    """
    PlantLatentDataset class that stores metadata about the image time series of one plant.
    Yields randomly sampled subsequences as inputs and targets in form of latent variables and their corresponding times.

    Args:
        metadata (Metadata): The metadata about the time series of one plant
        timespan (int): The range in which to sample the subsequences
        n_inputs (int): Length of the input sequence
        n_targets (int): Length of the target sequence
    """
    scale = 0.18215

    def __init__(self,
                 latents_dir: Path | str,
                 metadata: Metadata,
                 timespan: int,
                 n_inputs: int,
                 n_targets: int):
        latents_dir = Path(latents_dir)
        metadata['latent_path'] = metadata.latent_path.apply(lambda p: latents_dir/p)
        super(PlantLatentDataset, self).__init__(metadata, timespan, n_inputs, n_targets)

    def __getitem__(self, idx: int):
        x, y = super(PlantLatentDataset, self).__getitem__(idx)
        t_in = torch.tensor(x.time.to_numpy())
        t_out = torch.tensor(y.time.to_numpy())
        z_in = torch.stack([torch.load(p) for p in x.latent_path])
        z_out = torch.stack([torch.load(p) for p in y.latent_path])
        z_in = DiagonalGaussianDistribution(z_in).sample() * PlantLatentDataset.scale
        z_out = DiagonalGaussianDistribution(z_out).sample() * PlantLatentDataset.scale
        return z_in, t_in, t_out, z_out