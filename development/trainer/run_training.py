"""Module for training and validating the model."""
import re
from datetime import datetime
from pathlib import Path

import cv2
import torch.cuda
import wandb
from torch.utils.data import DataLoader

from development.data_io import dataloader
from development.data_io.dataloader import SizeLoader
from development.data_io.dataloader2 import PersonDataset, TestDataSet
from development.model.comb_model import CombModel
from development.trainer import level_manager
from development.trainer.trainer import Trainer
from development.trainer.training_file_io import TrainingIO
from development.trainer.visualizer import TrainVisualizer
from development.trainer.training_logger import WandBLogger


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


def get_test_set() -> DataLoader:
    """Get a generic dataset with only 100 samples."""
    return DataLoader(
        TestDataSet(
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
    loaders = [get_test_set(), get_test_set()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombModel(persons=len(loaders))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # manager = level_manager.ScoreGatedLevelManager(rate=blend_rate, min_score=0.95, max_level=8)
    manager = level_manager.LinearManager(rate=blend_rate, max_level=8)

    logger = WandBLogger(
        project="combmodel",
        entity="cglukas",
        learning_rate=learning_rate,
        blend_rate=blend_rate,
        optimizer=str(type(optimizer)),
    )

    start = datetime.now().strftime("%Y-%m-%d_%H_%M")
    filepath = (
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\trainings")
            / f"{start}"
    )
    if not filepath.exists():
        print(f"{filepath} created")
        filepath.mkdir(exist_ok=True)
    file_io = TrainingIO(model,optimizer,manager)
    file_io.set_folder(filepath)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloaders=loaders,
        file_io=file_io,
        device=device,
        logger=logger,
        level_manager=manager,
    )
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
