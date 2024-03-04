from pathlib import Path

import torch

from development.model.comb_model import CombModel


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
