"""Tests for the decoder neuronal network."""
import pytest
import torch

from development.data_io.dataloader2 import ImageSize
from development.model.decoder import Decoder


@pytest.mark.parametrize(("level", "size"), enumerate(ImageSize))
def test_decoder(level: int, size: int):
    """Test that all decoder levels reconstruct the latent vector to the input size."""
    rand = torch.ones((1, 512, 1, 1))
    dec = Decoder()

    reconstructed = dec.forward(rand, level=level)

    assert (1, 3, size, size) == reconstructed.shape


@pytest.mark.parametrize(("level", "size"), enumerate(ImageSize))
@pytest.mark.parametrize("last_lvl_influence", [0.0, 0.5, 1.0])
def test_decoder_progressive_forward(level: int, size: int, last_lvl_influence: float):
    """Test that the progressive forward reconstructs the input image from the latent vector."""
    rand = torch.ones((1, 512, 1, 1))
    dec = Decoder()

    reconstructed = dec.progressive_forward(
        rand, level=level, last_lvl_influence=last_lvl_influence
    )

    assert (1, 3, size, size) == reconstructed.shape
