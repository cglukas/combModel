"""Module for training and validating the model."""
import re
from pathlib import Path

import cv2
import torch.cuda
import wandb
from torch.utils.data import DataLoader

from development.data_io import dataloader
from development.data_io.dataloader import SizeLoader
from development.data_io.dataloader2 import PersonDataset
from development.model.comb_model import CombModel
from development.trainer.training import TrainLogger, TrainVisualizer, Trainer


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


def get_generic() -> DataLoader:
    """Get a dataloader for the general human faces dataset."""
    return DataLoader(
        PersonDataset(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\person3"),
        ),
        batch_size=8,
        shuffle=True,
    )


def main():
    """Main training routine."""
    ### Hyper parameter
    learning_rate = 10e-4  # 10e-4 is used in the disney research paper.
    blend_rate = 0.05
    bruce = DataLoader(
        PersonDataset(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\bruce"),
        ),
        batch_size=8,
        shuffle=True,
    )
    michael = DataLoader(
        PersonDataset(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\michael"),
        ),
        batch_size=8,
        shuffle=True,
    )
    loaders = [bruce, michael]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombModel(persons=len(loaders), device=device)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    logger = TrainLogger(
        learning_rate=learning_rate, blend_rate=blend_rate, optimizer=str(type(optimizer))
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloaders=loaders,
        device=device,
        max_level=4,
        logger=logger
    )
    trainer.blend_rate = blend_rate
    trainer.train()
    wandb.finish()


def validate(model_state_dict: Path | str) -> None:
    """Load the model and process a sample batch and display it for visual validation.

    Args:
        model_state_dict: path to the state dict of the trained model.
    """
    match = re.match(r".*_(?P<level>\d)-(?P<blend>\d\.\d)\.pth", str(model_state_dict))
    level = int(match.group("level"))
    blend = float(match.group("blend"))
    loaders = [get_generic(), get_generic()]
    dataloader.SizeLoader.scale = dataloader.SCALES[level]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombModel(persons=len(loaders), device=device)
    model.eval()
    model.to(device)
    state_dict = torch.load(model_state_dict)
    model.load_state_dict(state_dict)
    visualizer = TrainVisualizer()
    for person, loader in enumerate(loaders):
        for sample in loader:
            image, _ = sample
            image = image.to(device)
            with torch.no_grad():
                processed = model.progressive_forward(
                    person=person, batch=image, level=level, last_level_influence=blend
                )
            visualizer.add_batches(image, processed)
            break
    visualizer.show()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
