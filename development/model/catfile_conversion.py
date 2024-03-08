"""Conversion process to generate a nuke compliant cat file."""
from pathlib import Path

import torch
from torch import nn

from development.data_io.dataloader2 import ImageSize
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


def convert_model(model_checkpoint: Path, export_path: Path) -> None:
    """Load the checkpoint and convert the trained model to a catfile for nuke.

    Args:
        model_checkpoint: path of the model state dict.
        export_path: path where the catfile should be written to.
    """
    state_dict = torch.load(str(model_checkpoint))
    keys = state_dict.keys()
    num_persons = 0
    while f"decoders.{num_persons}.levels.0.0.weight" in keys:
        num_persons += 1

    model = CombModel(persons=num_persons)

    model.load_state_dict(state_dict)
    export_path.write_text("test")


def _convert_model_to_torchscript(model: NukeModel, temp_path: Path) -> None:
    """Convert the model to the intermediate torchscript file type.

    Args:
        model: loaded NukeModel.
        temp_path: export path for the torchscript file.
    """
    model.eval()
    width = height = ImageSize.from_index(model.level)
    trace_input = torch.rand(1, 3, height, width)
    # TODO use torch.jit.script instead of trace.
    #  this is only possible once the CombModel is scriptable.
    traced = torch.jit.trace(model, trace_input)
    traced.save(temp_path)
