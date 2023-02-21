import logging
import os
from code.data_io.dataloader import SCALES, SizeLoader
from code.model.comb_model import CombModel
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
os.environ["WANDB_MODE"] = "offline"


def train():
    wandb.init(project="compmodel", entity="cglukas")

    training_folder = Path(datetime.now().strftime("%y%m%d_%H%M%S"))
    training_folder.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_persons = 3
    model = CombModel(device=device, persons=max_persons)
    # model.load_state_dict(torch.load(r"C:\Users\Lukas\PycharmProjects\combModel\code\trainer\230220_204956\model2_0.66.pth"))
    # model.to(device)

    batch_size = 5
    loader_0 = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=max_persons,
            person=0,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    loader_1 = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=max_persons,
            person=1,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    loader_2 = DataLoader(
        SizeLoader(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
            max_persons=max_persons,
            person=2,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    previews = [None for _ in range(max_persons)]

    metric = StructuralSimilarityIndexMeasure()
    metric.to(device)

    # Hyper parameters:
    learning_rate = 1e-5
    blend_rate = 0.01
    dataset_size = 20

    logging_epoch = 10
    max_level = 6
    epoch = -1
    level = 0
    blend = 0.0

    optimizer = AdaBelief(model.parameters(), lr=learning_rate)
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "blend_rate": blend_rate,
            "optimizer": type(optimizer),
            "max_level": max_level,
            "additional info": "only one batch",
        }
    )
    cv2.namedWindow("Deepfake Preview")
    while True:
        epoch += 1
        metric.reset()
        running_loss = 0
        blend += blend_rate
        if blend > 1:
            blend = 0
            level += 1
            if level > max_level:
                blend = 1
        blend = min(1, blend)
        level = min(level, max_level)
        SizeLoader.scale = SCALES[level]
        for samples_0, samples_1, samples_2 in zip(loader_0, loader_1, loader_2):
            optimizer.zero_grad()

            samples_0 = samples_0.to(device)
            old_0 = samples_0
            inferred = model.progressive_forward(0, samples_0, level, blend)
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
            inferred = model.progressive_forward(1, samples_1, level, blend)
            if samples_1.shape[-1] < 16:
                samples_1 = torch.nn.functional.interpolate(samples_1, (16, 16))
                inferred = torch.nn.functional.interpolate(inferred, (16, 16))
            score = -metric(samples_1, inferred)
            score.backward()
            running_loss += score.item()
            optimizer.step()
            previews[1] = (samples_1[:3], inferred[:3])

            optimizer.zero_grad()
            samples_2 = samples_2.to(device)
            old_2 = samples_2
            inferred = model.progressive_forward(1, samples_2, level, blend)
            if samples_2.shape[-1] < 16:
                samples_2 = torch.nn.functional.interpolate(samples_2, (16, 16))
                inferred = torch.nn.functional.interpolate(inferred, (16, 16))
            score = -metric(samples_2, inferred)
            score.backward()
            running_loss += score.item()
            optimizer.step()
            previews[2] = (samples_2[:3], inferred[:3])

        running_score = -running_loss / dataset_size

        if epoch % logging_epoch == logging_epoch - 1:
            logging.info(
                f"Epoch {epoch}> Score {running_score}, Level {level}, blend {blend}"
            )
            with torch.no_grad():
                model.eval()
                swaps = []
                for index, (_samples, _inferred) in enumerate(previews):
                    if index == 0:
                        swap = model.progressive_forward(1, old_0[:3], level, blend)
                    else:
                        swap = model.progressive_forward(0, old_1[:3], level, blend)
                    if swap.shape[-1] < 16:
                        swap = torch.nn.functional.interpolate(swap, (16, 16))
                    swaps.append((_samples[:3], swap))
                model.train()

            all_previews = []
            for a, b in previews + swaps:
                all_previews.append(a)
                all_previews.append(b)

            pt_contact = torchvision.utils.make_grid(torch.cat(all_previews))
            preview_image = pt_contact.permute(1, 2, 0).detach().cpu().numpy()
            preview_image = cv2.cvtColor(preview_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Deepfake Preview", preview_image)
            cv2.waitKey(10)
            image = wandb.Image(pt_contact, caption="Reconstruction of model")
            wandb.log(
                {
                    "score": running_score,
                    "level": level,
                    "blend": blend,
                    "epoch": epoch,
                    "batch_size": samples_0.shape[0],
                    "output": image,
                }
            )
            wandb.watch(model)

            torch.save(
                model.state_dict(),
                str(training_folder / f"model{level}_{round(blend,2)}.pth"),
            )


if __name__ == "__main__":
    train()
