"""Tests for the dataloader2 module."""
import tempfile
from pathlib import Path

import pytest
import torch
import torchvision.io

from development.data_io.dataloader2 import ImageSize, PersonDataset


def test_image_size_from_level_index():
    """Test that the image size can be retrieved from the level index."""
    assert ImageSize.from_index(0) == ImageSize.LEVEL0
    assert ImageSize.from_index(6) == ImageSize.LEVEL6
    assert ImageSize.from_index(8) == ImageSize.LEVEL8


class TestPersonDataset:
    """Tests for the PersonDataset.

    It's necessary that the dataset can load all the images in the folder.
    and it should load the images of the current scale.
    """

    @pytest.fixture
    def dataset_3(self) -> str:
        """Create a dummy folder with a dataset size of 3."""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            scale_folder = folder / "scale_4"
            scale_folder.mkdir()
            for i in range(3):
                file = scale_folder / f"{i}.jpg"
                torchvision.io.write_jpeg(
                    torch.zeros((3, 4, 4), dtype=torch.uint8), str(file)
                )
            yield folder

    @pytest.fixture
    def dataset_0(self) -> str:
        """Create a dummy folder with a dataset size of 0."""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            scale_folder = folder / "scale_4"
            scale_folder.mkdir()
            yield folder

    def test_dataset_length(self, dataset_0: str, dataset_3: str):
        """Test that the dataset length can be fetched."""
        assert len(PersonDataset(dataset_0)) == 0
        assert len(PersonDataset(dataset_3)) == 3

    def test_dataset_entry(self, dataset_3: str):
        """Test that a single dataset entry consists out of a mask and an image."""
        image, matte = PersonDataset(dataset_3)[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(matte, torch.Tensor)
