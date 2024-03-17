"""Tests for the training configuration."""
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

import pytest

from development.trainer.run_training import (
    run_training_for_single_config,
)
from development.trainer.configured_training.load_from_config import ConfigError
from development.trainer.configured_training.configuration import TrainingConfig


# this pylint comment is necessary for file specific fixtures.
# pylint: disable=redefined-outer-name


def test_run_training_wrong_config():
    """Test that resume and pretrain checkpoint together will raise an error."""
    conf = TrainingConfig(resume_checkpoint="test", pretraining_checkpoint="test")
    with pytest.raises(
            ConfigError,
            match="Resuming with pretrained checkpoint does not work. Only provide one value.",
    ):
        run_training_for_single_config(conf)


@pytest.fixture()
def all_mocks() -> dict[str, MagicMock]:
    """Get a mock for all functions from the load_from_config."""
    with patch.multiple(
            "development.trainer.run_training",
            get_optimizer=DEFAULT,
            init_model_and_optimizer=DEFAULT,
            load_datasets=DEFAULT,
            load_level_manager=DEFAULT,
            load_logger=DEFAULT,
            CombModel=DEFAULT,
            Trainer=DEFAULT,
            TrainingIO=DEFAULT,
    ) as mocks:
        yield mocks


@pytest.mark.parametrize(
    "attribute",
    [
        "get_optimizer",
        "init_model_and_optimizer",
        "load_datasets",
        "load_level_manager",
        "load_logger",
        "CombModel",
        "Trainer",
        "TrainingIO",
    ],
)
def test_all_mock(all_mocks: dict[str, MagicMock], attribute: str) -> None:
    """Test that all attributes that are part of the training routine are mocked."""
    # pylint: disable=import-outside-toplevel
    from development.trainer import run_training as run_training_test

    # pylint: enable=import-outside-toplevel

    mocked_function = getattr(run_training_test, attribute)

    assert isinstance(mocked_function, MagicMock)
    assert mocked_function is all_mocks[attribute]


@pytest.fixture(autouse=True)
def mock_path() -> dict[str, MagicMock]:
    """Mock the path library to prevent file creation."""
    with patch.multiple(Path, mkdir=DEFAULT, exists=DEFAULT) as mocks:
        yield mocks


class TestModelInitialization:
    """Tests for the model initialisation"""

    def test_no_checkpoints(self, all_mocks: dict[str, MagicMock]) -> None:
        """Test that an untrained model is initialized when no checkpoint is provided.."""
        conf = TrainingConfig(datasets=[1, 2, 3])

        run_training_for_single_config(conf)

        all_mocks["init_model_and_optimizer"].assert_not_called()
        all_mocks["CombModel"].assert_called_with(persons=len(conf.datasets))
        all_mocks["get_optimizer"].assert_called_with(
            conf, all_mocks["CombModel"].return_value
        )
        all_mocks["TrainingIO"].return_value.load.assert_not_called()

    def test_pretrained_checkpoint(self, all_mocks: dict[str, MagicMock]):
        """Test that the pretrained checkpoint will be used."""
        conf = TrainingConfig(datasets=[1, 2, 3], pretraining_checkpoint="test.pth")
        all_mocks["init_model_and_optimizer"].return_value = (MagicMock(), MagicMock())

        run_training_for_single_config(conf)

        all_mocks["init_model_and_optimizer"].assert_called_with(conf)
        all_mocks["CombModel"].assert_not_called()
        all_mocks["get_optimizer"].assert_not_called()

    def test_resume_checkpoint(self, all_mocks: dict[str, MagicMock]):
        """Test that the pretrained checkpoint will be used."""
        conf = TrainingConfig(datasets=[1, 2, 3], resume_checkpoint="test.pth")

        run_training_for_single_config(conf)

        all_mocks["init_model_and_optimizer"].assert_not_called()
        all_mocks["CombModel"].assert_called_with(persons=len(conf.datasets))
        all_mocks["get_optimizer"].assert_called_with(
            conf, all_mocks["CombModel"].return_value
        )
        all_mocks["TrainingIO"].return_value.load.assert_called_with(Path("test.pth"))

    @pytest.mark.usefixtures("all_mocks")
    def test_resume_and_pretraining_used(self) -> None:
        """Test that a config error is raised when the pretraining and the resume checkpoint are used together."""
        config = TrainingConfig(
            resume_checkpoint="path1", pretraining_checkpoint="path2"
        )

        with pytest.raises(
                ConfigError,
                match="Resuming with pretrained checkpoint does not work. Only provide one value.",
        ):
            run_training_for_single_config(config)


def test_training_no_datasets() -> None:
    """Test that training without datasets fails."""
    with pytest.raises(ConfigError, match="No datasets provided."):
        run_training_for_single_config(TrainingConfig())


@pytest.mark.parametrize("exists", [True, False])
@pytest.mark.usefixtures("all_mocks")
def test_training_folder_handling(exists: bool, mock_path: dict[str, MagicMock]) -> None:
    """Test that the training folder will be created before training."""
    config = TrainingConfig(datasets=["1"], trainings_folder="Test")
    mock_path["exists"].return_value = exists

    run_training_for_single_config(config)

    mock_path["exists"].assert_called()
    assert mock_path["mkdir"].called != exists


