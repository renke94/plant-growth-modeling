from typing import Type

import lightning as L
from torch.utils.data import DataLoader

from .dataset_combine import DatasetCombine
from .metadata import Metadata
from .utils import batch_dim_collate_factory


class PlantDataModule(L.LightningDataModule):
    """
    Lightning DataModule for image time series datasets of Arabitopsis Thaliana.
    To provide a generic API, the dataset type can be passed, so that the data module can be used with different types
    of datasets that depend on the metadata.

    Args:
        metadata (Metadata): Metadata DataFrame about the whole dataset
        dataset_type (Type): Type of datasets to be instantiated
        train_size (int): number of plants used for training
        val_size (int): number of plants used for validation
        test_size (int): number of plants used for testing
        batch_dim (int): dimension the dataloaders have to stack the samples to a batch
        batch_size (int): batch size
        num_workers (int): number of workers for data loading (increase to prevent loading performance bottlenecks)
        random_seed (int): Optional random seed if the dataset split should be reconstructable
        **dataset_kwargs (dict): additional keyword arguments for dataset initialization
    """
    def __init__(self,
                 metadata: Metadata,
                 dataset_type: Type,
                 train_size: int,
                 val_size: int,
                 test_size: int,
                 batch_dim: int = 1,
                 batch_size: int = 8,
                 num_workers: int = 0,
                 random_seed: int | None = None,
                 **dataset_kwargs):
        super().__init__()
        self.ds_type = dataset_type
        self.metadata = metadata
        self.dataset_kwargs = dataset_kwargs
        self.split_sizes = [train_size, val_size, test_size]
        self.random_seed = random_seed
        self.dataloader_args = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=batch_dim_collate_factory(batch_dim)
        )

    def setup(self, stage=None):
        train_ds, val_ds, test_ds = self.metadata.split(self.split_sizes, self.random_seed)
        self.train_ds = DatasetCombine([self.ds_type(metadata=df, **self.dataset_kwargs) for df in train_ds.plants()])
        self.val_ds = DatasetCombine([self.ds_type(metadata=df, **self.dataset_kwargs) for df in train_ds.plants()])
        self.test_ds = DatasetCombine([self.ds_type(metadata=df, **self.dataset_kwargs) for df in train_ds.plants()])

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.dataloader_args | {'shuffle': True})

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.dataloader_args)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.dataloader_args)