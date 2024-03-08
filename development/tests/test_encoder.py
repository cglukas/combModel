"""Tests for the encoder neuronal network."""
import pytest
import torch

from development.data_io.dataloader2 import ImageSize
from development.model.encoder import Encoder


@pytest.mark.slow
@pytest.mark.parametrize(("level", "size"), enumerate(ImageSize))
def test_encoder(level: int, size: int):
    """Test that all encoder levels are encoded to the same latent tensor."""
    rand = torch.ones((1, 3, size, size))
    enc = Encoder()

    latent = enc.forward(rand, level=level)

    assert (1, 512, 1, 1) == latent.shape


@pytest.mark.slow
@pytest.mark.parametrize(("level", "size"), enumerate(ImageSize))
@pytest.mark.parametrize("last_lvl_influence", [0.0, 0.5, 1.0])
def test_encoder_progressive_forward(level: int, size: int, last_lvl_influence: float):
    """Test that the progressive forward encodes the input to the latent vector."""
    rand = torch.ones((1, 3, size, size))
    enc = Encoder()

    latent = enc.progressive_forward(
        rand, level=level, last_lvl_influence=last_lvl_influence
    )

    assert (1, 512, 1, 1) == latent.shape


@pytest.mark.xfail(reason="The encoder is currently not scriptable.")
def test_encoder_torchscript():
    """Test that the encoder can be scripted."""
    enc = Encoder()

    scripted = torch.jit.script(enc)

    assert isinstance(scripted, torch.jit.ScriptModule)
