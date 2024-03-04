"""Dataloader to load different image sizes"""
import warnings
from pathlib import Path

import PIL.Image
import torch
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset

SCALES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]


class SizeLoader(Dataset):
    """Dataset for the images of persons.

    This is a unified dataset for all persons. Multiple instances can be used to load individual
    persons, but it will always consider the dataset length of the smallest set.
    """

    scale = 4
    # person = 0

    def __init__(self, preprocessed_path: Path, max_persons: int = 0, person: int = 0):
        warnings.warn(
            "Do not use SizeLoader anymore. The design of this class is very complicated "
            "and not easy to extend/maintain. Use the PersonDataset instead."
        )
        self.filenames = {}
        self.preprocessed_path = preprocessed_path
        self.max_persons = max_persons
        self.load_filenames()
        self.transforms = torchvision.transforms.ToTensor()
        self.index = 0
        self.person = person

    def __len__(self) -> int:
        """The size of the smallest available dataset.

        It's necessary to use the smallest dataset because else we can't be sure
        that the item access can load a sample.
        """
        lengths = [len(fn) for fn in self.filenames.values()]
        return min(lengths)

    def __getitem__(self, index) -> torch.Tensor:
        """Get one image sample of the dataset."""
        self.index = index
        return self.load_sample()

    def load_filenames(self):
        """Load all images for all persons to create the image index."""
        for person in range(self.max_persons):
            reference_scale = (
                self.preprocessed_path / f"person{person}" / f"scale_{self.scale}"
            )
            self.filenames[person] = sorted(reference_scale.iterdir())

    def load_batch(
        self,
        starting_index: int,
        level: int,
        max_batchsize: int = 40,
        max_mb: float = 0.3,
    ) -> torch.Tensor:
        """Load one batch of images.

        Notes:
            If max_batchsize and max_mb are used, it will only satisfy one of them.
            If the batchsize is reached before the mb, then it will return a batch
            with less memory usage. If the max_mb is reached before the batchsize,
            then the batch will contain fewer images.

        Args:
            starting_index: the image index for the first sample in the batch.
            level: the image level that should be loaded.
            max_batchsize: the maximum amount of images in the batch.
            max_mb: the maximum byte size of the batch.

        Returns:
            Batch of image tensors (shape: [batchsize, channels, width, height]).
        """
        self.scale = SCALES[level]
        self.index = starting_index
        max_bytes = max_mb * 1e6
        batch = torch.Tensor()
        sample_size = self._get_size(self.load_sample())
        while (
            self._get_size(batch) + sample_size < max_bytes
            and batch.shape[0] < max_batchsize
        ):
            batch = torch.cat([batch, self.load_sample()])
            self.index += 1
        return batch

    @staticmethod
    def _get_size(tensor: torch.Tensor) -> int:
        """Get the byte size of the tensor."""
        return tensor.nelement() * tensor.element_size()

    def load_sample(self) -> torch.Tensor:
        """Load one image and preprocess it with the configured transforms.

        Returns:
            Tensor of the image.
        """
        img_path = self.get_filepath()
        img = PIL.Image.open(str(img_path))
        tensor = self.transforms(img)
        return tensor

    def get_filepath(self):
        """Get the filepath for scaled image of the selected person.

        Returns:
            filepath to the image.
        """
        filename = self.filenames[self.person][self.index].name
        return (
            self.preprocessed_path
            / f"person{self.person}"
            / f"scale_{self.scale}"
            / filename
        )


def test_size_loader():
    """Test the size loader."""
    loader = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=1,
            person=0,
        ),
        batch_size=30,
        shuffle=True,
    )
    batch = next(iter(loader))
    assert batch.shape == (30, 3, 4, 4)

    SizeLoader.scale = 8
    batch = next(iter(loader))
    assert batch.shape == (30, 3, 8, 8)
