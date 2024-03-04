"""Visualize latent space."""
from pathlib import Path

import torch.cuda
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from development.data_io.dataloader import SCALES, SizeLoader
from development.data_io.dataloader2 import ImageSize, PersonDataset
from development.model.comb_model import CombModel


def display_latent_encoding(
    model_path: str, datasets: list[PersonDataset], level: int, max_samples: int = -1
):
    """Display the latent space of the model for the samples of the datasets.

    Notes:
        You can process datasets with non-matching sizes by setting the max_samples to the size of the smallest dataset.

    Args:
        model_path: path to the comb model checkpoint.
        datasets: list of datasets. All datasets need to be of the same size.
        level: model level that should be used for processing.
        max_samples: maximum of samples to process. Default -1 will use all samples of dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO remove once PersonLoader has no dependencies to this anymore.
    SizeLoader.scale = SCALES[level]

    model = CombModel(device=device, persons=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Algorithm to reduce dimensionality of latent vectors.
    # This will produce 2D points for plotting.
    tsne = TSNE(2)
    _, ax = plt.subplots()
    for person, loader in enumerate(datasets):
        loader.set_scale(ImageSize.from_index(level))
        latents = []
        for index, (sample, _) in enumerate(loader):
            # Second part of loader will be the mask. This is only necessary for training.
            batch = sample[None, :]

            model.progressive_forward(
                person, batch.to(device), level, last_level_influence=1
            )
            latents.append(model.latent.detach().cpu())
            if max_samples != -1 and index > max_samples:
                break

        latent_2d = tsne.fit_transform(torch.cat(latents).squeeze())
        ax.scatter(*zip(*latent_2d), alpha=0.5, label=f"Person {person}")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    display_latent_encoding(
        model_path=r"C:\Users\Lukas\PycharmProjects\combModel\trainings\27-02-24_19_05\comb_model_2-0.8.pth",
        datasets=[
            PersonDataset(
                Path(
                    r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\person3"
                ),
            )
        ]
        * 2,
        level=1,
        max_samples=1000,
    )
