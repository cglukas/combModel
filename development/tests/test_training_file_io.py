"""Tests for handling model and optimizer files during training."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from torch.optim import Optimizer

from development.model.comb_model import CombModel
from development.trainer.level_manager import LinearManager
from development.trainer.training_file_io import TrainingIO


@pytest.fixture(name="model_and_optim_mock")
def fixture_model_and_optim_mock() -> tuple[MagicMock, MagicMock]:
    """Get a model and optimizer as magic mocks"""
    model = MagicMock(spec=CombModel)
    model.state_dict.return_value = {"model": 1}

    optimizer = MagicMock(spec=Optimizer)
    optimizer.state_dict.return_value = {"optimizer": 1}

    return model, optimizer


@patch("torch.save")
def test_save(save_patch: MagicMock, model_and_optim_mock: tuple[MagicMock, MagicMock]):
    """Test that the save method will call torch.save with the model state and the expected filepath."""
    model, optimizer = model_and_optim_mock

    manager = LinearManager(rate=0.4)
    manager.level = 2
    manager.blend = 0.5
    test_folder = Path("testfolder")
    expected_model_file = test_folder / "model_2_0.5.pth"
    expected_optimizer_file = test_folder / "optim_2_0.5.pth"

    # This is the call to test:
    io_handler = TrainingIO(model, optimizer, manager)
    io_handler.save(test_folder)

    save_patch.assert_any_call(model.state_dict(), str(expected_model_file))
    save_patch.assert_any_call(optimizer.state_dict(), str(expected_optimizer_file))


@patch("torch.load")
def test_load(load_patch: MagicMock, model_and_optim_mock: tuple[MagicMock, MagicMock]):
    """Test that the load method will call torch.load.

    It's also important that the saved files will be used to load the states for the model and optimizer.
    The level manager also needs to be updated with the used level and blend.
    """
    model, optimizer = model_and_optim_mock
    load_patch.return_value = {"state": "dict"}

    manager = LinearManager(rate=0.4)
    manager.level = 0
    manager.blend = 0.0
    test_folder = Path("testfolder")
    expected_model_file = test_folder / "model_2_0.5.pth"
    expected_optimizer_file = test_folder / "optim_2_0.5.pth"

    # This is the call to test:
    io_handler = TrainingIO(model, optimizer, manager)
    io_handler.load(expected_model_file)

    # Assertions:
    assert manager.level == 2
    assert manager.blend == 0.5

    load_patch.assert_any_call(str(expected_model_file))
    load_patch.assert_any_call(str(expected_optimizer_file))

    model.load_state_dict.assert_called_with({"state": "dict"})
    optimizer.load_state_dict.assert_called_with({"state": "dict"})
