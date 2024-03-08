"""Tests for the comb model."""
import pytest
import torch

from development.data_io.dataloader2 import ImageSize
from development.model.comb_model import (
    CombModel,
)


@pytest.mark.parametrize(("level", "size"), enumerate(ImageSize))
def test_comb_model(level: int, size: int):
    """Test that the expected tensor sizes will be processed correctly."""
    rand = torch.ones((1, 3, size, size))
    comb = CombModel()

    reconstruct_comb = comb.progressive_forward(0, rand, level, 0.5)

    assert rand.shape == reconstruct_comb.shape
