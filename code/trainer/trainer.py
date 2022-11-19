from pathlib import Path

import cv2
import torch
import torchvision.utils
from torch.optim import Adam

from code.data_io.dataloader import SizeLoader, SCALES
from code.model.comb_model import CombModel
from torchmetrics import StructuralSimilarityIndexMeasure
from adabelief_pytorch import AdaBelief


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombModel(device=device)
    model.to(device)
    dataloader = SizeLoader(
        Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\person1")
    )
    metric = StructuralSimilarityIndexMeasure()
    metric.to(device)
    optimizer = AdaBelief(model.parameters(), lr=1e-5)
    level = 0
    blend = 0.0
    grow_rate = 1e-3
    epoch = -1
    while True:
        epoch += 1
        metric.reset()
        running_loss = 0
        blend += grow_rate
        if blend > 1:
            blend = 0
            level += 1

        for index in range(10):
            optimizer.zero_grad()
            samples = dataloader.load_batch(index, level, max_batchsize=40, max_mb=3)
            samples = samples.to(device)
            inferred = model.progressive_forward(0, samples, level, 0)
            if samples.shape[-1] < 16:
                samples = torch.nn.functional.interpolate(samples, (16, 16))
                inferred = torch.nn.functional.interpolate(inferred, (16, 16))
            score = -metric(samples, inferred)
            score.backward()
            running_loss += score.item()
            optimizer.step()
        print(f"Epoch {epoch}> Score {-running_loss/10}, Level {level}, blend {blend}")
        contact = torchvision.utils.make_grid(torch.cat([samples[:3], inferred[:3]]))
        contact = contact.permute(1, 2, 0).detach().cpu().numpy()

        cv2.imshow("Training", contact)
        cv2.waitKey(10)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model{level}.pth")


if __name__ == "__main__":
    train()
