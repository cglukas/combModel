"""Model"""
import torch
from torch import nn

from development.model.decoder import Decoder
from development.model.encoder import Encoder


class CombModel(nn.Module):
    """Implementation of the comb model from the disney research paper.

    This model allows parallel training of multiple identities that will be encoded in the same latent space.

    Paper:
        https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/
    """

    def __init__(self, persons: int = 2):
        super().__init__()
        self.encoder = Encoder()
        self.decoders = nn.ModuleList()
        self.latent: torch.Tensor
        for _ in range(persons):
            decoder = Decoder()
            self.decoders.append(decoder)

    def forward(self):
        """Use progressive_forward."""
        raise NotImplementedError

    def progressive_forward(
        self, person: int, batch: torch.Tensor, level: int, last_level_influence: float
    ) -> torch.Tensor:
        """Process the batch in the forward pass through the encoder and corresponding decoder.

        Args:
            person: index of the person decoder.
            batch: batch of images [batchsize, channels, width, height].
            level: determines which subset of the encoder/decoder is used (0-8).
            last_level_influence: blending factor to blend between the smaller level and the current level.

        Returns:
            processed batch with reconstructed images.
        """
        decoder = self.decoders[person]
        self.latent = self.encoder.progressive_forward(
            batch, level, last_lvl_influence=last_level_influence
        )
        return decoder.progressive_forward(
            self.latent, level, last_lvl_influence=last_level_influence
        )
