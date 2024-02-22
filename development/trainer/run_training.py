import re
from pathlib import Path

import cv2
import torch.cuda
from torch.utils.data import DataLoader

from development.data_io.dataloader import SizeLoader
from development.model.comb_model import CombModel
from development.trainer.training import TrainVisualizer, Trainer


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
    return DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=5,
            person=4,
        ),
        batch_size=8,
        shuffle=True,
    )


def main():
    loaders = _get_loaders(batch_size=8)
    generic = get_generic()
    loaders = [generic, generic]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombModel(persons=len(loaders), device=device)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=10e-4, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloaders=loaders,
        device=device,
        max_level=2,
    )
    trainer.blend_rate = 10e-4
    # trainer.current_level = 2
    # trainer.current_blend = 0.2
    trainer.train()


def validate(model_state_dict: Path | str):

    re.match(r".*_(?P<level>\d)-(?P<blend>\d\.\d)\.pth", str(model_state_dict))
    loaders = _get_loaders(batch_size=2)
    loaders = [loaders[2], loaders[2]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombModel(persons=len(loaders), device=device)
    model.to(device)
    model.load_state_dict(torch.load(model_state_dict))
    visualizer = TrainVisualizer()
    for person, loader in enumerate(loaders):
        for image in loader:
            visualizer.add_image(image)
            processed = model.progressive_forward(
                person=person, tensor=image, level=level, last_level_influence=blend
            )
            visualizer.add_image(processed)
    visualizer.show()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
    # validate(r"C:\Users\Lukas\PycharmProjects\combModel\trainings\05-07-23_08_49\comb_model_1-1.0.pth")
