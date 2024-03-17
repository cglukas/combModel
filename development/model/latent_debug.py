"""Visualize latent space."""
from pathlib import Path

import numpy
import torch.cuda
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from development.data_io.dataloader2 import ImageSize, PersonDataset
from development.model.comb_model import CombModel
from development.model.utils import initialize_comb_model


def display_latent_encoding(model_path: str, datasets: list[PersonDataset], level: int):
    """Display the latent space of the model for the samples of the datasets.

    Notes:
        You can process datasets with non-matching sizes by setting the max_samples to the size of the smallest dataset.

    Args:
        model_path: path to the comb model checkpoint.
        datasets: list of datasets. All datasets need to be of the same size.
        level: model level that should be used for processing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = initialize_comb_model(Path(model_path))
    model.to(device)

    all_latents = []
    for person, dataset in enumerate(datasets):
        latent = compute_latent_vectors(model, dataset, person=person, level=level)
        latent_2d = reduce_dimension(latent)
        all_latents.append(latent_2d)
    labels = [f"Person{i}" for i in range(len(datasets))]
    generate_figure(all_latents, labels=labels)
    plt.show()


def reduce_dimension(latent_space: torch.Tensor) -> torch.Tensor:
    """Reduce the tensor dimension to 2d.

    Notes:
        This function uses TSNE which tries to compress the structure of the
        high dimensional data down to 2d. It's not perfect, as it's loosing
        a lot of information in this process.

    Args:
        latent_space: high dimension tensor. shape: (num samples, 512, 1, 1)

    Returns:
        Tensor with shape (num samples, 2)
    """
    if latent_space.shape < (2, 512, 1, 1):
        msg = "Tensor needs to be of shape (samples, 512, 1, 1) where samples are greater than 1."
        raise ValueError(msg)
    samples = latent_space.shape[0]
    tsne = TSNE(n_components=2, perplexity=min(samples * 0.5, 30))
    squeeze = latent_space.squeeze()
    return tsne.fit_transform(squeeze)


def compute_latent_vectors(
    model: CombModel, dataset: PersonDataset, person: int, level: int
) -> torch.Tensor:
    """Compute the latent vectors for each sample in the dataset.

    Args:
        model: model to use for processing.
        dataset: datasamples to process.
        person: the person index that corresponds to the dataset.
        level: the image size level that should be used for processing.

    Returns:
        The combined latent vectors for the full dataset (shape: (samples, 512, 1, 1)).
    """
    latents = []
    dataset.set_scale(ImageSize.from_index(level))
    for sample, _ in tqdm(dataset):
        batch = sample[None, :]
        model.progressive_forward(
            person=person, batch=batch, level=level, last_level_influence=1
        )
        latents.append(model.latent.detach().cpu())

    return torch.cat(latents)


def generate_figure(points: list[torch.Tensor], labels: list[str] = None) -> plt.Figure:
    """Generate a figure with the points in a scatter plot.

    Args:
        points: list of sequences of points (shape (samples, 2)).
        labels: list of labels for the point sequences.

    Returns:
        Created figure.
    """
    fig, ax = plt.subplots()
    labels = labels or [""] * len(points)
    for samples, label in zip(points, labels):
        ax.scatter(*numpy.transpose(samples), alpha=0.5, label=label)
    ax.legend()
    return fig


if __name__ == "__main__":
    device = torch.device("cuda")
    datasets = [
        PersonDataset(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\bruce"),
            device=device,
        ),
        PersonDataset(
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed\michael"),
            device=device,
        ),
    ]
    display_latent_encoding(
        model_path=r"C:\Users\Lukas\PycharmProjects\combModel\trainings\2024-03-15_08_55\model_5_1.0.pth",
        datasets=datasets,
        level=2,
    )
