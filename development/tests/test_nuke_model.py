"""Tests for the nuke model."""
import torch

from development.model.comb_model import CombModel
from development.model.nuke_model import NukeModel


def test_nuke_model_forward() -> None:
    """Test that the forward method returns a tensor of the same size."""
    model = CombModel()
    test_input = torch.zeros(1, 3, 128, 128)

    processed = NukeModel(model).forward(test_input)

    assert processed.shape == test_input.shape
