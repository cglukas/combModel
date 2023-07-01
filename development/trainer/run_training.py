from pathlib import Path

import torch.cuda
from torch.utils.data import DataLoader

from development.data_io.dataloader import SizeLoader
from development.trainer.training import Trainer


def _get_loaders(batch_size: int) -> list[DataLoader]:
    max_persons = 3
    loader_0 = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=max_persons,
            person=0,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    loader_1 = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=max_persons,
            person=1,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    loader_2 = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=max_persons,
            person=2,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return [loader_0, loader_1, loader_2]


def main():
    loaders = _get_loaders(batch_size=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(dataloaders=loaders, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
