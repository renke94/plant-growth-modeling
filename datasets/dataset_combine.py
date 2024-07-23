from typing import Iterable, Type

import numpy as np
from torch.utils.data import Dataset

from .metadata import Metadata


class DatasetCombine(Dataset):
    """
    Helper class to combine multiple datasets whose samples should be distinct.

    Args:
        subsets (Iterable[Dataset]): Collection of datasets to combine.
    """
    def __init__(self, subsets: Iterable[Dataset]):
        self.subsets = subsets
        self.subset_lengths = np.array([len(ds) for ds in self.subsets])
        self.index_map = np.cumsum(self.subset_lengths)

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        subset_idx = binary_search(self.index_map, idx)
        sub_idx = idx - ([0] + list(self.index_map))[subset_idx]
        sequence = self.subsets[subset_idx][sub_idx]
        return sequence


def binary_search(array, value):
    def bs(start, end, idx):
        if end - start < 0:
            return idx
        i = (start + end) // 2
        v = array[i]
        if value >= v:
            return bs(i+1, end, idx)
        elif value < v:
            return bs(start, i-1, i)

    return bs(0, len(array), -1)


def search(array, value):
    for i in range(0, len(array)):
        if value < array[i]:
            return i
    return -1


class MultiplePlantDataset(DatasetCombine):
    def __init__(self,
                 metadata: Metadata,
                 dataset_type: Type,
                 **dataset_kwargs):
        super(MultiplePlantDataset, self).__init__([
            dataset_type(metadata=plant, **dataset_kwargs)
            for plant in metadata.plants()
        ])
