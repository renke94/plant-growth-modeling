import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import crop, to_pil_image
from torch import Tensor, nn
from tqdm import tqdm

from ..models.lgm import LatentGrowthVAE
from .metadata import Metadata
from .utils import rgbvi, load_image

tray_grids = [
    {
        'x': [380, 830, 1290, 1740, 2190],
        'y': [310, 750, 1210, 1650],
    },
    {
        'x': [380, 830, 1290, 1740, 2170],
        'y': [310, 750, 1210, 1640],
    },
    {
        'x': [380, 830, 1290, 1740, 2200],
        'y': [310, 750, 1210, 1650],
    },
    {
        'x': [400, 850, 1290, 1740, 2180],
        'y': [300, 740, 1190, 1635],
    },
]

class TrayCropper:
    """
    Helper class to crop the single plants from a whole tray.

    The center points of the plants are transferred in the form of a grid consisting of a list of x-coordinates and
    y-coordinates. The number of resulting crops results from the number of intersection points in the grid:
    len(grid_x) * len(grid_y)

    Args:
        crop_size (int): square size of the crops.
        grid_x (list[int]): list of x-coordinates of the grid
        grid_y (list[int]): list of y-coordinates of the grid
    """
    def __init__(self, crop_size: int, grid_x: list[int], grid_y: list[int]):
        s = crop_size // 2
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.crop = lambda img, x, y: crop(img, y - s, x - s, crop_size, crop_size)

    def __call__(self, img: Tensor) -> Tensor:
        """
        Crops the given image and returns a tensor containing the cropped image.

        Args:
            img (Tensor): image to be cropped
        Returns:
            Tensor of shape [len(grod_y), len(grid_x), C, image_size, image_size] containing the cropped images.
        """
        return torch.stack([
            torch.stack([self.crop(img, x, y) for x in self.grid_x])
            for y in self.grid_y
        ])


class RGBVIFilter:
    """
    Module to separate the foreground and background of images from the Aberystwyth Leaf Evaluation dataset.
    The separation includes two steps:
    1. creating a binary mask using RGBVI scores exceeding the rgbvi_threshold
    2. blurring the mask to smooth the mask and thresholding it again

    RGB images are converted to RGBA images containing the binary mask as alpha channel.

    Args:
        rgbvi_threshold (float): 0 <= rgbvi_threshold <= 1
        blur_kernel_size (int): size of the Gaussian blurring kernel
        blur_sigma (float | Tuple[float, float]): standard deviation of the Gaussian blurring can be either
            fixed or a range from which sigma is uniformly chosen.
        blur_threshold (float): threshold to binarize the mask after blurring
    """
    def __init__(self,
                 rgbvi_threshold: float = 0.20,
                 blur_kernel_size: int = 13,
                 blur_sigma: Union[float, tuple[float, float]] = 4.0,
                 blur_threshold: float = 0.5):
        self.blur = T.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
        self.blur_threshold = blur_threshold
        self.rgbvi_threshold = rgbvi_threshold

    def __call__(self, rgb_images: Tensor) -> Tensor:
        """
        Creates a binary mask separating foreground and background using RGBVI scores for separation and Gaussian
        blurring to smooth out the mask.

        Args:
            rgb_images (Tensor): RGB images in form of tensors, where each tensor must at least have the three image
            dimensions [..., C, H, W].

        Returns:
            RGBA images containing the binary mask as alpha channel.
        """
        mask = rgbvi(rgb_images) > self.rgbvi_threshold
        mask = self.blur(mask.float()) > self.blur_threshold
        return torch.cat([rgb_images, mask.float()], dim=-3)


def timestamp_from_path(path: Path) -> int:
    date = datetime.strptime(path.stem.split('_')[2], '%Y-%m-%d--%H-%M-%S')
    return int(date.timestamp())

class TrayPreprocessor:
    """
    Preprocessor for a single tray of the Aberystwyth Leaf Evaluation Dataset.
    A single tray is a directory, that contains time series images capturing the growth of 20 Arabitopsis plants.

    The source directory images are cropped into multiple images. The crops are labeled column-wise with a b c ... and
    row-wise with 1 2 3 ...  In addition, the corresponding timestamps are stored in file names. The images identified
    by the positions a1, ... d4 are also saved in corresponding folders so that each plant is located in a separate
    folder.

    Example:
        Name of the original image: PSI_Tray031_2015-12-14--12-54-06_top.png
        Path to the first crop: a1/1-a1-1450094046.png

    Args:
        source_dir (Path): Directory containing the time series images.
        target_dir (Path): Directory in which to save the cropped images.
        tray_number (int): Number of the tray (needed for naming the cropped images).
        crop_size (int): Size of the images to be cropped.
        grid (dict): Dictionary of grid coordinates. These coordinates define the center points of the plants.
        Must have the following keys:
            'x': List[int]
            'y': List[int]
        rgbvi_filter (RGBVIFilter): A pre-configured RGBVIFilter object, to separate foreground and background of the
        images.
    """
    def __init__(self,
                 source_dir: Path,
                 target_dir: Path,
                 tray_number: int,
                 crop_size: int,
                 grid: dict,
                 rgbvi_filter: RGBVIFilter):
        self.tray_number = tray_number
        self.source_dir = source_dir
        self.source_files = list(source_dir.glob('*.png'))
        self.target_dir = target_dir
        self.tray_cropper = TrayCropper(crop_size, grid['x'], grid['y'])
        self.rgbvi_filter = rgbvi_filter
    def process_files(self):
        """
        Starts the preprocessing of the tray.
        """
        for filename in tqdm(self.source_files, desc=f'Preprocessing {self.tray_number}', file=sys.stdout):
            tray_image = load_image(filename)
            tray_image = self.rgbvi_filter(tray_image)  # separate foreground and background
            images = self.tray_cropper(tray_image)      # make image crops

            # save images
            for i, row in enumerate(images):
                for j, image in zip("abcdefghijklmnop", row):
                    target_path = self.target_dir / f"{j}{i}/{self.tray_number}-{j}{i}-{timestamp_from_path(filename)}.png"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    to_pil_image(image).save(target_path)


class ImageToLatentDataset(Dataset):
    """
    Dataset that is used for the conversion of images to latent representations.
    The dataset yields an image with its corresponding latent path.

    Args:
        metadata (Metadata): Metadata DataFrame containing the images paths and their corresponding latent paths.
    """
    def __init__(self, metadata: Metadata):
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image = load_image(row.path)
        return image, row.latent_path


class ImageToLatentConverter(nn.Module):
    """
    The ImageToLatentConverter converts images to their latent representations using the pre-trained AutoencoderKL
    from the Diffusers library.

    The purpose of this module is to pre-calculate the latent variables to avoid the overhead of computing them each
    time they are needed.

    Args:
        metadata (Metadata): Metadata DataFrame containing the images paths and their corresponding latent paths.
        batch_size (int): Batch size for the dataloader
        num_workers (int): Number of workers for multiprocess data loading.
        transforms (callable, optional): transforms performed on the images before they are fed into the encoder. Here
        augmentations, rescaling, and more can be applied.
    """
    def __init__(self, metadata: Metadata, batch_size: int = 16, num_workers: int = 0, transforms=None):
        super(ImageToLatentConverter, self).__init__()
        self.dataset = ImageToLatentDataset(metadata)

        def image_to_latent_collate_fn(batch) -> tuple:
            images, paths = zip(*batch)
            images = torch.stack(images)
            return images, paths

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=image_to_latent_collate_fn
        )
        self.vae = LatentGrowthVAE()
        self.transforms = transforms

    @torch.no_grad()
    def convert_images(self):
        """
        Converts the images to latent representations under the given configuration
        """
        print("Creating directories")
        directories = self.dataset.metadata.latent_path.apply(lambda p: p.parent).unique()
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)

        device = next(self.vae.parameters()).device
        for batch in tqdm(self.dataloader, desc='Converting images to latents', file=sys.stdout):
            images, latent_paths = batch
            images = images.to(device)
            if self.transforms is not None:
                images = self.transforms(images)
            latents = self.vae.vae.encode(images * 2 - 1).latent_dist.parameters.cpu()
            for latent, path in zip(latents, latent_paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(latent, path)

