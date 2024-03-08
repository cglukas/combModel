import pytest
import torch

from development.model.encoder import Encoder
from test_comb_model import _levels


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