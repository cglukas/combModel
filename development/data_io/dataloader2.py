from __future__ import annotations
import enum
from functools import lru_cache
from pathlib import Path

import PIL.Image
import torch
import torchvision
from torch.utils.data import Dataset

from development.data_io.dataloader import SizeLoader


class IMAGE_SIZE(enum.IntEnum):
    Level0 = 4
    Level1 = 8
    Level2 = 16
    Level3 = 32
    Level4 = 64
    Level5 = 128
    Level6 = 256
    Level7 = 512
    Level8 = 512

    @classmethod
    def from_index(cls, index) -> IMAGE_SIZE:
        all_levels = [
            cls.Level0,
            cls.Level1,
            cls.Level2,
            cls.Level3,
            cls.Level4,
            cls.Level5,
            cls.Level6,
            cls.Level7,
            cls.Level8,
        ]
        return all_levels[index]


class PersonDataset(Dataset):
    def __init__(self, folder: str | Path):
        """Initialize the PersonDataset.

        Args:
            folder: folder containing the subfolders of scaled images.
        """
        self._folder = Path(folder)
        self._transforms = torchvision.transforms.ToTensor()
        self._scale = IMAGE_SIZE.Level0

    def set_scale(self, scale: IMAGE_SIZE) -> None:
        """Set the scale that should be loaded."""
        self._scale = scale

    @lru_cache
    def _get_images_for_scale(self,  scale: IMAGE_SIZE) -> list[Path]:
        """Get all image files for the required image scale.

        Returns:
            list of found images.
        """
        scale_folder = self._folder / f"scale_{scale.value}"
        if not scale_folder.exists() or not scale_folder.is_dir():
            raise OSError(f"Can't find the folder: '{scale_folder}'")
        return list(scale_folder.iterdir())

    @lru_cache
    def _get_mattes_for_scale(self, scale: IMAGE_SIZE) -> list[Path]:
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
        self._scale = IMAGE_SIZE(SizeLoader.scale)

        img_files = self._get_images_for_scale(self._scale)
        img_path = img_files[index]
        img = PIL.Image.open(str(img_path))
        matte_files = self._get_mattes_for_scale(self._scale)
        transformed_img = self._transforms(img)
        if matte_files:
            matte_path = matte_files[index]
            matte = PIL.Image.open(str(matte_path))
            return transformed_img, self._transforms(matte)
        return transformed_img, torch.ones(transformed_img.shape)
