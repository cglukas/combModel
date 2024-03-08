import pytest
import torch

from development.model.decoder import Decoder
from test_comb_model import _levels


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