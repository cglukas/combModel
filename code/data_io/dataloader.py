"""Dataloader to load different image sizes"""
from pathlib import Path

import PIL.Image
import torch
import torchvision.transforms

SCALES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

class SizeLoader:
    def __init__(self, preprocessed_path: Path):
        self.filenames = []
        self.preprocessed_path = preprocessed_path
        self.load_filenames()
        self.transforms = torchvision.transforms.ToTensor()
        self.scale = 4
        self.index = 0

    def load_filenames(self):
        reference_scale = self.preprocessed_path / "scale_4"
        self.filenames = sorted(reference_scale.iterdir())

    def load_batch(
        self, starting_index: int, level: int, max_batchsize: int = 40, max_mb: float = 0.3
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
        tensor = tensor[None, :]
        return tensor

    def get_filepath(self):
        filename = self.filenames[self.index].name
        return (
                self.preprocessed_path / f"scale_{self.scale}" / filename
        )



if __name__ == "__main__":
    loader = SizeLoader(
        Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\person1")
    )
    batch = loader.load_batch(0, 256, max_mb=10)
    print(batch.shape)
