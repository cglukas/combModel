"""Tests for the model utilities."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from development.model.comb_model import CombModel
from development.model.utils import (
    initialize_comb_model,
    initialize_comb_model_from_pretraining,
)


@pytest.mark.parametrize("num_persons", [1, 2, 4])
def test_initialize_comb_model(num_persons: int) -> None:
    """Test that the model will be loaded with the right amount of persons."""
    model = CombModel(persons=num_persons)

    with TemporaryDirectory() as export_dir:
        export_dir = Path(export_dir)
        state_dict = export_dir / "state_dict.pth"
        torch.save(model.state_dict(), state_dict)

        loaded_model = initialize_comb_model(state_dict)

        assert len(loaded_model.decoders) == num_persons


@pytest.mark.parametrize("num_persons", [1, 2, 4])
def test_initialize_comb_model_from_pretraining(num_persons: int) -> None:
    """Test that a model with different persons can be initialized from the state dict of pretrained one."""
    pretrained = CombModel(persons=1)

    with TemporaryDirectory() as export_dir:
        export_dir = Path(export_dir)
        state_dict = export_dir / "state_dict.pth"
        torch.save(pretrained.state_dict(), state_dict)

        loaded_model = initialize_comb_model_from_pretraining(state_dict, num_persons)

    assert len(loaded_model.decoders) == num_persons

    reference_decoder = pretrained.decoders[0]
    for decoder in loaded_model.decoders:
        for p1, p2 in zip(decoder.parameters(), reference_decoder.parameters()):
            assert (
                p1.data.ne(p2.data).sum() == 0
            ), "Some of the parameters are not equal."
