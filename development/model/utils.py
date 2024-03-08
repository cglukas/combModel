"""Utilities for the comb model."""
from pathlib import Path

import torch

from development.model.comb_model import CombModel


def initialize_comb_model(model_checkpoint: Path) -> CombModel:
    """Initialize a comb model instance from the checkpoint.

    Args:
        model_checkpoint: file of the checkpoint.

    Returns:
        Initialized model with the right amount of persons from the checkpoint.
    """
    state_dict = torch.load(str(model_checkpoint))
    keys = state_dict.keys()
    num_persons = 0
    while f"decoders.{num_persons}.levels.0.0.weight" in keys:
        num_persons += 1
    model = CombModel(persons=num_persons)
    model.load_state_dict(state_dict)
    return model
