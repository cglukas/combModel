"""Test the latent debug module."""
import re
from unittest.mock import ANY, MagicMock

import matplotlib.pyplot as plt
import pytest
import torch

from development.data_io.dataloader2 import ImageSize, PersonDataset
from development.model.comb_model import CombModel
from development.model.latent_debug import (
    compute_latent_vectors,
    generate_figure,
    reduce_dimension,
)


@pytest.mark.parametrize("n_samples", [2, 12, 50])
def test_reduce_dimensions(n_samples: int) -> None:
    """Test that a multidimensional tensor can be reduced to 2d."""
    tensor = torch.rand((n_samples, 512, 1, 1))

    reduced = reduce_dimension(tensor)

    assert reduced.shape == (n_samples, 2)


@pytest.mark.parametrize(
    "wrong_input", [(1, 1), (1, 512, 1, 1), (0, 512, 1, 1), (2, 2, 1, 1)]
)
def test_reduce_dimension_wrong_input(wrong_input: tuple) -> None:
    """Test that a value error is raised when a wrong tensor size is used."""
    tensor = torch.rand(wrong_input)

    msg = "Tensor needs to be of shape (samples, 512, 1, 1) where samples are greater than 1."
    with pytest.raises(ValueError, match=re.escape(msg)):
        reduce_dimension(tensor)


def test_get_latent_vectors() -> None:
    """Test that a tensor of latent spaces is returned."""
    model = MagicMock(wraps=CombModel(persons=2))
    model.latent = torch.ones((1, 512, 1, 1))
    dataset = MagicMock(autospec=PersonDataset)
    dataset.__iter__.return_value = iter(
        [(torch.ones(3, 32, 32), torch.ones(3, 32, 32))] * 4
    )

    latent = compute_latent_vectors(model, dataset=dataset, person=0, level=3)

    assert latent.shape == (4, 512, 1, 1)
    model.progressive_forward.assert_called_with(
        person=0, batch=ANY, level=3, last_level_influence=1
    )
    dataset.set_scale.assert_called_with(ImageSize.from_index(3))


def test_generate_figure() -> None:
    """Test that a figure can be generated from a list of 2d vectors."""
    latents = [torch.rand(12, 2), torch.rand(21, 2)]

    figure = generate_figure(points=latents, labels=["Test1", "test"])

    assert isinstance(figure, plt.Figure)
