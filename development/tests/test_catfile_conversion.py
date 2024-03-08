"""Tests for the catfile conversion process."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
import torch

from development.model.catfile_conversion import (
    NukeModel,
    _convert_model_to_torchscript,
    convert_model,
)
from development.model.comb_model import CombModel


def test_nuke_model_forward() -> None:
    """Test that the forward method returns a tensor of the same size."""
    model = CombModel()
    test_input = torch.zeros(1, 3, 128, 128)

    processed = NukeModel(model).forward(test_input)

    assert processed.shape == test_input.shape


@patch("torch.load", wraps=torch.load)
def test_convert_model(torch_load: MagicMock):
    """Test that the model can be converted from the state dict file."""
    model = CombModel()
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        state_dict = tmp_dir / "state_dict.pth"
        torch.save(model.state_dict(), state_dict)
        export_file = tmp_dir / "export.pt"

        convert_model(state_dict, export_file)

        torch_load.assert_called_with(str(state_dict))
        exported_model = torch.jit.load(export_file)
    expected_model = NukeModel(model)
    assert exported_model.state_dict().keys() == expected_model.state_dict().keys()


@pytest.mark.slow
@pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_convert_model_to_torchscript(level: int) -> None:
    """Test that the nuke model can be converted to torchscript."""
    model = CombModel(persons=2)
    nuke_model = NukeModel(model)
    nuke_model.level = level

    with TemporaryDirectory() as export_dir:
        model_file = Path(export_dir) / "test_export.pt"
        _convert_model_to_torchscript(nuke_model, model_file)

        assert model_file.exists()
