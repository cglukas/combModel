"""Utilities for the comb model."""
import re
from collections import OrderedDict
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


def initialize_comb_model_from_pretraining(pretraining_checkpoint: Path, num_persons: int) -> CombModel:
    """Initialize a comb model with the weights from the pretraining for all decoders.

    Args:
        pretraining_checkpoint: path to the state dict with trained weights and biases.
        num_persons: amount of persons you want to train. The initialized model will contain this amount of decoders.

    Returns:
        Initialized model.
    """
    decoder_re = re.compile(r"decoders\.(?P<person>\d+)\.")
    state_dict = torch.load(str(pretraining_checkpoint))

    new_dict = OrderedDict()
    for key, value in state_dict.items():
        if decoder_re.match(key):
            for i in range(num_persons):
                n_key = decoder_re.sub(f"decoders.{i}.", key)
                new_dict[n_key] = torch.clone(value)
        else:
            new_dict[key] = value

    model = CombModel(persons=num_persons)
    model.load_state_dict(new_dict)
    return model
