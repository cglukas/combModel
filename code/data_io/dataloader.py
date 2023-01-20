"""Dataloader to load different image sizes"""
from pathlib import Path

import PIL.Image
import torch
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset

SCALES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]


class SizeLoader(Dataset):
    scale = 4
    # person = 0

    def __init__(self, preprocessed_path: Path, max_persons:int = 0, person: int = 0):
        self.filenames = {}
        self.preprocessed_path = preprocessed_path
        self.max_persons = max_persons
        self.load_filenames()
        self.transforms = torchvision.transforms.ToTensor()
        self.index = 0
        self.person = person

    def __len__(self):
        lengths = [len(fn) for fn in self.filenames.values()]
        return min(lengths)

    def __getitem__(self, index):
        self.index = index
        return self.load_sample()

    def load_filenames(self):
        for person in range(self.max_persons):
            reference_scale = self.preprocessed_path / f"person{person}" / f"scale_{self.scale}"
            self.filenames[person] = sorted(reference_scale.iterdir())

    def load_batch(
        self,
        starting_index: int,
        level: int,
        max_batchsize: int = 40,
        max_mb: float = 0.3,
    ) -> torch.Tensor:
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

    def _get_size(self, tensor: torch.Tensor):
        return tensor.nelement() * tensor.element_size()

    def load_sample(self) -> torch.Tensor:
        img_path = self.get_filepath()
        img = PIL.Image.open(str(img_path))
        tensor = self.transforms(img)
        return tensor

    def get_filepath(self):
        filename = self.filenames[self.person][self.index].name
        return self.preprocessed_path / f"person{self.person}" / f"scale_{self.scale}" / filename


if __name__ == "__main__":
    loader = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed")
        ),
        batch_size=30,
        shuffle=True,
    )
    batch = next(iter(loader))
    print("30, 4x4", batch.shape)

    SizeLoader.scale = 8
    batch = next(iter(loader))
    print("30, 8x8", batch.shape)

