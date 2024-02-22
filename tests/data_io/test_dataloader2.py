import tempfile
from pathlib import Path

import pytest
import torch
import torchvision.io

from development.data_io.dataloader2 import IMAGE_SIZE, PersonDataset


class TestIMAGE_SIZE:
    def test_init_from_number(self):
        assert IMAGE_SIZE.from_index(0) == IMAGE_SIZE.Level0
        assert IMAGE_SIZE.from_index(8) == IMAGE_SIZE.Level8


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
        return

    @pytest.fixture
    def dataset_0(self) -> str:
        """Create a dummy folder with a dataset size of 0."""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            scale_folder = folder / "scale_4"
            scale_folder.mkdir()
            yield folder
        return

    def test_dataset_length(self, dataset_0: str, dataset_3: str):
        assert len(PersonDataset(dataset_0)) == 0
        assert len(PersonDataset(dataset_3)) == 3

    def test_dataset_entry(self, dataset_3: str):
        image, matte = PersonDataset(dataset_3)[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(matte, torch.Tensor)
