"""Conversion process to generate a nuke compliant cat file."""
import os
import sys
from pathlib import Path

import click
import torch

from development.data_io.dataloader2 import ImageSize
from development.model.nuke_model import NukeModel
from development.model.utils import initialize_comb_model


def convert_model(model_checkpoint: Path, export_path: Path, level: int = 8) -> None:
    """Load the checkpoint and convert the trained model to a catfile for nuke.

    Args:
        model_checkpoint: path of the model state dict.
        export_path: path where the catfile should be written to.
        level: image level of the model for evaluation. (Default uses LEVEL 8)
    """
    model = initialize_comb_model(model_checkpoint)
    nuke_model = NukeModel(model)
    nuke_model.level = level
    _convert_model_to_torchscript(nuke_model, export_path)


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


@click.command()
@click.option("--model", "-m", prompt=True, type=str)
@click.option("--export", "-e", prompt=True, type=str)
@click.option("--level", "-lvl", type=click.IntRange(0, 8), default=8)
def _parse_args(model: str, export: str, level: int) -> int:
    """Convert a model checkpoint into a torchscript file."""
    export_file = Path(export)
    if not os.access(export_file.parent, os.W_OK):
        print(f"Can't write to export path: '{export_file}'")
        return 1

    model_checkpoint = Path(model)
    if not model_checkpoint.exists():
        print(f"Checkpoint can't be found: '{model_checkpoint}'.")
        return 1

    convert_model(model_checkpoint, export_file, level)
    if not export_file.exists():
        print("Something went wrong while exporting.")
        return 1

    print("Finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(_parse_args())  # pylint: disable=E1120
