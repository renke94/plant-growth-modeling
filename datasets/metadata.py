from pathlib import Path
from typing import Optional

import pandas as pd
import torch


class Metadata(pd.DataFrame):
    """
    Metadata class extending pandas DataFrame by helpful methods regarding metadata management of the Arabitopsis
    dataset.
    """
    # start of recordings
    min_timestamp = 1450097646

    @staticmethod
    def process_path(path: Path) -> dict:
        parts = path.stem.split('-')
        if len(parts) != 3:
            raise ValueError(f'Invalid metadata path: {path}')
        tray, pos, time = parts
        hour = (int(time) - Metadata.min_timestamp) // 3600
        plant = f"{tray}-{pos}"
        latent_path = Path(*path.parts[-3:-1], f"{plant}-{time}.pt")
        return {'plant': plant, 'latent_path': latent_path, 'time': hour}

    @staticmethod
    def from_directory(directory: Path | str) -> pd.DataFrame:
        """

        Args:
            directory (Path): directory containing the cropped images

        Returns:
            Instance of type Metadata which is a dataframe containing the paths of the images and their corresponding
            plant and time labels. In addition, a latent path is provided for direct access to corresponding latent
            variables
        """
        image_files = list(Path(directory).rglob('*.png'))
        image_files = pd.DataFrame({'path': image_files})
        metadata = image_files.path.apply(Metadata.process_path).to_list()
        metadata = pd.DataFrame(metadata)
        metadata = pd.concat([image_files, metadata], axis=1)
        metadata.__class__ = Metadata
        return metadata

    @staticmethod
    def load(csv_path: Path | str) -> pd.DataFrame:
        """
        Loads a csv file and returns it as a Metadata instance.
        The csv file must have the following columns:
        - path (str) -> Path
        - plant (str) -> str
        - latent_path (str) -> Path
        - time (int) -> int

        Where the '->' indicates the type in which the columns are converted

        Args:
            csv_path (Path): path to the metadata csv file

        Returns:
            Instance of type Metadata
        """
        metadata = pd.read_csv(csv_path, converters={
            'path': Path,
            'latent_path': Path
        }, index_col=0)
        metadata.__class__ = Metadata
        return metadata


    def save(self, csv_path: Path | str):
        """
        Saves the metadata to a csv file at a given path.

        Args:
            csv_path (Path): Path where the metadata will be saved
        """
        self.to_csv(csv_path, index=False)

    def n_plants(self) -> int:
        """
        Returns:
             Number of unique plants contained in the dataset as int.
        """
        return self.groupby('plant').ngroups

    def plants(self) -> list[pd.DataFrame]:
        """
        Returns:
            A list of dataframes with each plant in a separate dataframe
        """
        return [g for _, g in self.groupby('plant')]

    def split(self, sizes: list[int], random_seed: int = None) -> list['Metadata']:
        """
        Splits the metadata into subsets based on the specified sizes and returns them as a list of dataframes.

        Args:
            sizes (list[int]): list of arbitrary length containing the sizes of the desired subsets
            random_seed (int | None): if not None the split is performed using the seed. Defaults to None

        Returns:
            list[pd.DataFrame]: Subsets of the metadata dataframe.
        """
        plants = self.plants()
        if random_seed is not None:
            torch.random.manual_seed(random_seed)
        perms = torch.randperm(len(plants)).split(sizes)
        perms = [pd.concat([plants[i] for i in p]) for p in perms]
        for p in perms:
            p.__class__ = Metadata
        return perms

    def query(self, expr: str, inplace: bool = False, **kwargs) -> Optional['Metadata']:
        """
        Override of pandas DataFrame.query and extends it by a cast to type Metadata.

        Query the columns of a Metadata with a boolean expression.

        Args:
            expr (str): The query string to evaluate.
            inplace (bool): Whether to perform the operation in-place

        Returns:
            Metadata resulting from the provided query expression or None if ``inplace=True``.

        You can refer to variables in the environment by prefixing them with an '@' character like ``@a + b``.

        You can refer to column names that are not valid Python variable names by surrounding them in backticks. Thus,
        column names containing spaces or punctuations (besides underscores) or starting with digits must be surrounded
        by backticks. (For example, a column named "Area (cm^2)" would be referenced as ```Area (cm^2)```).

        Column names which are Python keywords (like "list", "for", "import", etc) cannot be used. For example, if one
        of your columns is called ``a a`` and you want to sum it with ``b``, your query should be ```a a` + b``.
        """
        out = super(Metadata, self).query(expr=expr, inplace=inplace, **kwargs)
        if out is not None:
            out.__class__ = Metadata
        return out

    def remove_empty_pots(self):
        """
        Returns a version of this Metadata where all empty pots have been removed
        """
        return self.query("plant_exists == True")

    def train_ds(self) -> 'Metadata':
        """
        Returns a Metadata Object containing the training samples
        """
        return self.query("subset =='train'")


    def val_ds(self) -> 'Metadata':
        """
        Returns a Metadata Object containing the validation samples
        """
        return self.query("subset =='val'")


    def test_ds(self) -> 'Metadata':
        """
        Returns a Metadata Object containing the test samples
        """
        return self.query("subset =='test'")