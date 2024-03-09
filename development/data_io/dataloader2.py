"""Module for the person dataset loader."""
from __future__ import annotations
import enum
from functools import lru_cache
from pathlib import Path

import PIL.Image
import torch
import torchvision
from torch.utils.data import Dataset

from development.data_io.dataloader import SizeLoader


class ImageSize(enum.IntEnum):
    """All possible image sizes for the comb model levels."""

    LEVEL0 = 4
    LEVEL1 = 8
    LEVEL2 = 16
    LEVEL3 = 32
    LEVEL4 = 64
    LEVEL5 = 128
    LEVEL6 = 256
    LEVEL7 = 512
    LEVEL8 = 1024

    @classmethod
    def from_index(cls, index) -> ImageSize:
        """Get the enum for the level index."""
        all_levels = list(cls)
        return all_levels[index]


class PersonDataset(Dataset):
    """Dataset for loading a single person dataset for the comb model."""

    def __init__(self, folder: str | Path, device: torch.device):
        """Initialize the PersonDataset.

        Args:
            folder: folder containing the subfolders of scaled images.
        """
        self._folder = Path(folder)
        self._transforms = torchvision.transforms.ToTensor()
        self._scale = ImageSize.LEVEL0
        self._device = device

    def set_scale(self, scale: ImageSize) -> None:
        """Set the scale that should be loaded."""
        self._scale = scale

    @lru_cache
    def _get_images_for_scale(self, scale: ImageSize) -> list[Path]:
        """Get all image files for the required image scale.

        Returns:
            list of found images.
        """
        scale_folder = self._folder / f"scale_{scale.value}"
        if not scale_folder.exists() or not scale_folder.is_dir():
            raise OSError(f"Can't find the folder: '{scale_folder}'")
        return list(scale_folder.iterdir())

    @lru_cache
    def _get_mattes_for_scale(self, scale: ImageSize) -> list[Path]:
        """Get all matte files for the required image scale.

        Returns:
            list of found images.
        """
        scale_folder = self._folder / f"scale_{scale.value}_matte"
        if not scale_folder.exists() or not scale_folder.is_dir():
            return []
        return list(scale_folder.iterdir())

    def __len__(self):
        """Length of the dataset sampled from the current image scale folder."""
        return len(self._get_images_for_scale(self._scale))

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single image of the dataset as a tensor.

        Args:
            index: index in the list of all images.

        Returns:
            Tensor of the loaded image.
        """
        # TODO remove this once the scale setting is implemented
        self._scale = ImageSize(SizeLoader.scale)

        img_files = self._get_images_for_scale(self._scale)
        img_path = img_files[index]
        img = PIL.Image.open(str(img_path))
        matte_files = self._get_mattes_for_scale(self._scale)
        transformed_img = self._transforms(img)
        if matte_files:
            matte_path = matte_files[index]
            matte = PIL.Image.open(str(matte_path))
            return transformed_img.to(self._device), self._transforms(matte).to(self._device)
        return transformed_img.to(self._device), torch.ones(transformed_img.shape).to(self._device)


class TestDataSet(PersonDataset):
    """Test dataset that only loads the first 100 images."""
    def __len__(self):
        return 100
