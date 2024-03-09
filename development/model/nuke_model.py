"""Module of the nuke model that is used for cat file conversion."""
import torch
from torch import nn

from development.model.comb_model import CombModel


class NukeModel(nn.Module):
    """Nuke implementation of the comb model.

    This class is used for converting the pytorch model to a 'catfile' that
    can be executed inside nuke.
    """

    def __init__(self, model: CombModel):
        super().__init__()
        self.model = model
        self.level = 1
        self.person = 1

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Process an input image with the loaded model.

        Args:
            image: tensor of the input image [1, channels, height, width]

        Returns:
            processed image as tensor [1, channels, height, width]
        """
        return self.model.progressive_forward(
            person=self.person, batch=image, level=self.level, last_level_influence=1
        )
