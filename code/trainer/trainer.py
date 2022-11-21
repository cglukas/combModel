from datetime import datetime
from pathlib import Path

import cv2
import torch
import torchvision.utils
import wandb
from torch.optim import Adam

from code.data_io.dataloader import SizeLoader, SCALES
from code.model.comb_model import CombModel
from torchmetrics import StructuralSimilarityIndexMeasure
from adabelief_pytorch import AdaBelief


def train():
    wandb.init(project="compmodel", entity="cglukas")

    training_folder = Path(datetime.now().strftime("%y%m%d_%H%M%S"))
    training_folder.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombModel(device=device)
    model.to(device)
    person_1 = SizeLoader(
        Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\person1")
    )
    person_2 = SizeLoader(
        Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\person2")
    )
    metric = StructuralSimilarityIndexMeasure()
    metric.to(device)
    # Hyper parameters:
    learning_rate = 1e-5
    blend_rate = 1e-4
    dataset_size = 10
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
            "additional info": "Use adaptive blend rate",
        }
    )
    while True:
        epoch += 1
        metric.reset()
        running_loss = 0
        blend += blend_rate - (level * adaptive_blend_rate * blend_rate)
        if blend > 1:
            blend = 0
            level += 1

        for index in range(dataset_size):
            optimizer.zero_grad()
            samples = person_1.load_batch(index, level, max_batchsize=40, max_mb=3)
            samples = samples.to(device)
            inferred = model.progressive_forward(0, samples, level, blend)
            if samples.shape[-1] < 16:
                samples = torch.nn.functional.interpolate(samples, (16, 16))
                inferred = torch.nn.functional.interpolate(inferred, (16, 16))
            score = -metric(samples, inferred)
            score.backward()
            running_loss += score.item()
            optimizer.step()

            optimizer.zero_grad()
            samples2 = person_2.load_batch(index, level, max_batchsize=40, max_mb=3)
            samples2 = samples2.to(device)
            inferred2 = model.progressive_forward(1, samples2, level, blend)
            if samples2.shape[-1] < 16:
                samples2 = torch.nn.functional.interpolate(samples2, (16, 16))
                inferred2 = torch.nn.functional.interpolate(inferred2, (16, 16))
            score = -metric(samples2, inferred2)
            score.backward()
            running_loss += score.item()
            optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}> Score {-running_loss/dataset_size}, Level {level}, blend {blend}"
            )
            model.eval()
            with torch.no_grad():
                swap = model.progressive_forward(0, samples2, level, blend)
                if swap.shape[-1] < 16:
                    swap = torch.nn.functional.interpolate(swap, (16, 16))

            model.train()

            pt_contact = torchvision.utils.make_grid(
                torch.cat(
                    [samples[:3], inferred[:3], samples2[:3], inferred2[:3], swap[:3]]
                )
            )

            image = wandb.Image(pt_contact, caption="Reconstruction of model")
            wandb.log(
                {
                    "score": -running_loss / dataset_size,
                    "level": level,
                    "blend": blend,
                    "epoch": epoch,
                    "batch_size": samples.shape[0],
                    "output": image,
                }
            )
            wandb.watch(model)

            torch.save(model.state_dict(), str(training_folder / f"model{level}.pth"))


if __name__ == "__main__":
    train()
