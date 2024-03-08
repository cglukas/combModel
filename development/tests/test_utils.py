"""Tests for the model utilities."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from development.model.comb_model import CombModel
from development.model.utils import initialize_comb_model


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
