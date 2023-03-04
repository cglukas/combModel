import logging
import os
from development.data_io.dataloader import SCALES, SizeLoader
from development.model.comb_model import CombModel
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torchvision.utils
import wandb
from adabelief_pytorch import AdaBelief
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
# os.environ["WANDB_MODE"] = "offline"


def train():
    wandb.init(project="compmodel", entity="cglukas")

    training_folder = Path(datetime.now().strftime("%y%m%d_%H%M%S"))
    training_folder.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombModel(device=device, persons=2)
    model.to(device)

    max_persons = 2

    loader_0 = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=max_persons,
            person=0
        ),
        batch_size=5,
        shuffle=True,
    )
    loader_1 = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=max_persons,
            person=1
        ),
        batch_size=5,
        shuffle=True,
    )

    previews = [None for _ in range(max_persons)]

    metric = StructuralSimilarityIndexMeasure()
    metric.to(device)

    logging_epoch = 50

    # Hyper parameters:
    learning_rate = 1e-5
    blend_rate = 1e-3
    dataset_size = 20
    adaptive_blend_rate = 0.125

    epoch = -1
    level = 0
    blend = 0.0

    optimizer = AdaBelief(model.parameters(), lr=learning_rate)
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "blend_rate": blend_rate,
            "adaptive_blend_rate": adaptive_blend_rate,
            "optimizer": type(optimizer),
            "additional info": "Use adaptive blend rate with squared level factor",
        }
    )
    while True:
        epoch += 1
        metric.reset()
        running_loss = 0
        blend += blend_rate * max(
            (1 - level ** 2 * adaptive_blend_rate), adaptive_blend_rate
        )
        if blend > 1:
            blend = 0
            level += 1
        SizeLoader.scale = SCALES[level]
        for samples_0, samples_1 in zip(loader_0, loader_1):
            optimizer.zero_grad()

            samples_0 = samples_0.to(device)
            old_0 = samples_0
            inferred = model.progressive_forward(
                0, samples_0, level, blend
            )
            if samples_0.shape[-1] < 16:
                samples_0 = torch.nn.functional.interpolate(samples_0, (16, 16))
                inferred = torch.nn.functional.interpolate(inferred, (16, 16))
            score = -metric(samples_0, inferred)
            score.backward()
            running_loss += score.item()
            optimizer.step()
            previews[0] = (samples_0[:3], inferred[:3])

            optimizer.zero_grad()
            samples_1 = samples_1.to(device)
            old_1 = samples_1
            inferred = model.progressive_forward(
                1, samples_1, level, blend
            )
            if samples_1.shape[-1] < 16:
                samples_1 = torch.nn.functional.interpolate(samples_1, (16, 16))
                inferred = torch.nn.functional.interpolate(inferred, (16, 16))
            score = -metric(samples_1, inferred)
            score.backward()
            running_loss += score.item()
            optimizer.step()
            previews[1] = (samples_1[:3], inferred[:3])


        if epoch % logging_epoch == logging_epoch - 1:
            logging.info(
                f"Epoch {epoch}> Score {-running_loss/dataset_size}, Level {level}, blend {blend}"
            )
            with torch.no_grad():
                model.eval()
                swaps = []
                for index, (_samples, _inferred) in enumerate(previews):
                    if index == 0:
                        swap = model.progressive_forward(
                           1 , old_0[:3], level, blend
                        )
                    else:
                        swap = model.progressive_forward(
                           0 , old_1[:3], level, blend
                        )
                    if swap.shape[-1] < 16:
                        swap = torch.nn.functional.interpolate(swap, (16, 16))
                    swaps.append((_samples[:3], swap))
                model.train(True)

            all_previews = []
            for a, b in previews + swaps:
                all_previews.append(a)
                all_previews.append(b)

            pt_contact = torchvision.utils.make_grid(torch.cat(all_previews))
            cv2.imshow("", pt_contact.permute(1, 2, 0).detach().cpu().numpy())
            cv2.waitKey(10)
            image = wandb.Image(pt_contact, caption="Reconstruction of model")
            wandb.log(
                {
                    "score": -running_loss / dataset_size,
                    "level": level,
                    "blend": blend,
                    "epoch": epoch,
                    "batch_size": samples_0.shape[0],
                    "output": image,
                }
            )
            wandb.watch(model)

            torch.save(model.state_dict(), str(training_folder / f"model{level}.pth"))


if __name__ == "__main__":
    train()
