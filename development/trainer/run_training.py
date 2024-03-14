"""Module for training and validating the model."""
import re
from datetime import datetime
from pathlib import Path

import cv2
import torch.cuda
import wandb
from adabelief_pytorch import AdaBelief

from development.data_io.dataloader2 import ImageSize, PersonDataset, TestDataSet
from development.data_io.dataset_manager import DatasetManager
from development.model.comb_model import CombModel
from development.trainer import level_manager
from development.trainer.trainer import Trainer
from development.trainer.training_file_io import TrainingIO
from development.trainer.visualizer import TrainVisualizer
from development.trainer.training_logger import WandBLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_generic() -> PersonDataset:
    """Get a dataloader for the general human faces dataset."""
    return PersonDataset(
        Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\person3"),
        device=DEVICE,
    )


def get_test_set() -> PersonDataset:
    """Get a generic dataset with only 100 samples."""
    return TestDataSet(
        Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\person3"),
        device=DEVICE,
    )


def main():
    """Main training routine."""
    ### Hyper parameter
    learning_rate = 10e-4  # 10e-4 is used in the disney research paper.
    blend_rate = 0.05
    # bruce = DataLoader(
    #     PersonDataset(
    #         Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\bruce"),
    #     ),
    #     batch_size=8,
    #     shuffle=True,
    # )
    # michael = DataLoader(
    #     PersonDataset(
    #         Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\michael"),
    #     ),
    #     batch_size=8,
    #     shuffle=True,
    # )
    datasets = [get_generic()]
    model = CombModel(persons=len(datasets))
    model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = adabelief_gan_small(model)
    lvl_manager = level_manager.ScoreGatedLevelManager(
        rate=blend_rate, min_score=0.95, max_level=8, max_repeat=10
    )
    # lvl_manager = level_manager.LinearManager(rate=blend_rate, max_level=8)
    dataset_manager = DatasetManager(datasets)
    logger = WandBLogger(
        project="combmodel",
        entity="cglukas",
        learning_rate=learning_rate,
        blend_rate=blend_rate,
        optimizer=str(type(optimizer)),
    )

    start = datetime.now().strftime("%Y-%m-%d_%H_%M")
    filepath = Path(r"C:\Users\Lukas\PycharmProjects\combModel\trainings") / f"{start}"
    if not filepath.exists():
        print(f"{filepath} created")
        filepath.mkdir(exist_ok=True)
    file_io = TrainingIO(model, optimizer, lvl_manager)
    file_io.set_folder(filepath)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataset_manager=dataset_manager,
        file_io=file_io,
        device=DEVICE,
        logger=logger,
        level_manager=lvl_manager,
    )
    trainer.train()
    wandb.finish()


def adabelief_gan_small(model) -> AdaBelief:
    """Get a AdaBelief instance for the model with the GAN(small) settings applied."""
    return AdaBelief(
        model.parameters(),
        lr=6e-5,
        betas=(0.5, 0.999),
        eps=1e-12,
        weight_decay=0,
        weight_decouple=True,
        rectify=False,
        fixed_decay=False,
    )


def validate(model_state_dict: Path | str) -> None:
    """Load the model and process a sample batch and display it for visual validation.

    Args:
        model_state_dict: path to the state dict of the trained model.
    """
    match = re.match(r".*_(?P<level>\d)-(?P<blend>\d\.\d)\.pth", str(model_state_dict))
    level = int(match.group("level"))
    blend = float(match.group("blend"))
    loaders = [get_generic(), get_generic()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombModel(persons=len(loaders))
    model.eval()
    model.to(device)
    state_dict = torch.load(model_state_dict)
    model.load_state_dict(state_dict)
    visualizer = TrainVisualizer()
    for person, loader in enumerate(loaders):
        loader.set_scale(ImageSize.from_index(level))
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
