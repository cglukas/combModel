import pytest
import torch

from development.model.comb_model import (
    CombModel,
)

_levels = list(enumerate([4, 8, 16, 32, 64, 128, 256, 512, 1024]))


@pytest.mark.parametrize(("level", "size"), _levels)
def test_comb_model(level: int, size: int):
    """Test that the expected tensor sizes will be processed correctly."""
    rand = torch.ones((1, 3, size, size))
    comb = CombModel()

    reconstruct_comb = comb.progressive_forward(0, rand, level, 0.5)

    assert rand.shape == reconstruct_comb.shape
