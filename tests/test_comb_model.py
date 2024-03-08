import pytest
import torch
from torch import nn

from development.model.comb_model import (
    CombModel,
    Decoder,
    Encoder,
    torchscript_index_access,
)


@pytest.mark.parametrize(
    ("num_modules", "index"),
    [
        (1, 0),
        (2, 0),
        (4, 2),
    ],
)
def test_torchscript_index_access(num_modules: int, index: int):
    """Test that the function replicates index accesses."""
    modules = [nn.ReLU() for _ in range(num_modules)]
    modules_list = nn.ModuleList(modules)

    found_module = torchscript_index_access(index=index, modules=modules_list)

    assert found_module == modules[index]


def test_torchscript_index_access_index_out_of_range():
    """Test that an index error is raised."""
    with pytest.raises(IndexError, match="Index not found in list."):
        torchscript_index_access(index=1, modules=nn.ModuleList())


_levels = list(enumerate([4, 8, 16, 32, 64, 128, 256, 512, 1024]))


@pytest.mark.parametrize(("level", "size"), _levels)
def test_comb_model(level: int, size: int):
    """Test that the expected tensor sizes will be processed correctly."""
    rand = torch.ones((1, 3, size, size))
    comb = CombModel()

    reconstruct_comb = comb.progressive_forward(0, rand, level, 0.5)

    assert rand.shape == reconstruct_comb.shape


class TestDecoder:
    @pytest.mark.parametrize(("level", "size"), _levels)
    def test_decoder(self, level: int, size: int):
        """Test that all decoder levels reconstruct the latent vector to the input size."""
        rand = torch.ones((1, 512, 1, 1))
        dec = Decoder()

        reconstructed = dec.forward(rand, level=level)

        assert (1, 3, size, size) == reconstructed.shape

    @pytest.mark.parametrize(("level", "size"), _levels)
    @pytest.mark.parametrize("last_lvl_influence", [0.0, 0.5, 1.0])
    def test_decoder_progressive_forward(
        self, level: int, size: int, last_lvl_influence: float
    ):
        """Test that the progressive forward reconstructs the input image from the latent vector."""
        rand = torch.ones((1, 512, 1, 1))
        dec = Decoder()

        reconstructed = dec.progressive_forward(
            rand, level=level, last_lvl_influence=last_lvl_influence
        )

        assert (1, 3, size, size) == reconstructed.shape


class TestEncoder:
    @pytest.mark.parametrize(("level", "size"), _levels)
    def test_encoder(self, level: int, size: int):
        """Test that all encoder levels are encoded to the same latent tensor."""
        rand = torch.ones((1, 3, size, size))
        enc = Encoder()

        latent = enc.forward(rand, level=level)

        assert (1, 512, 1, 1) == latent.shape

    @pytest.mark.parametrize(("level", "size"), _levels)
    @pytest.mark.parametrize("last_lvl_influence", [0.0, 0.5, 1.0])
    def test_encoder_progressive_forward(
        self, level: int, size: int, last_lvl_influence: float
    ):
        """Test that the progressive forward encodes the input to the latent vector."""
        rand = torch.ones((1, 3, size, size))
        enc = Encoder()

        latent = enc.progressive_forward(
            rand, level=level, last_lvl_influence=last_lvl_influence
        )

        assert (1, 512, 1, 1) == latent.shape

    def test_encoder_torchscript(self):
        """Test that the encoder can be scripted."""
        enc = Encoder()

        scripted = torch.jit.script(enc)

        assert isinstance(scripted, torch.jit.ScriptModule)
