"""Tests for the training configuration."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import adabelief_pytorch.AdaBelief
import pytest
import torch
from adabelief_pytorch import AdaBelief

from development.data_io.dataset_manager import DatasetManager
from development.model.comb_model import CombModel
from development.trainer.configured_training import (
    TrainingConfig,
    _init_model_and_optimizer,
    _load_datasets,
)


@pytest.mark.parametrize("datasets", [["test", "test2"], ["test"]])
@patch("development.trainer.configured_training.PersonDataset")
def test_load_datasets(dataset_mock: MagicMock, datasets: list[str]):
    """Test that the datasets are loaded into a dataset manager."""
    conf = TrainingConfig(datasets=datasets, device="cpu")
    device = torch.device("cpu")

    manager = _load_datasets(conf)

    assert isinstance(manager, DatasetManager)
    assert dataset_mock.call_count == len(datasets)
    for dataset in datasets:
        dataset_mock.assert_any_call(dataset, device=device)


def test_load_datasets_empty() -> None:
    """Test that empty dataset lists will raise an error."""
    conf = TrainingConfig()

    with pytest.raises(ValueError, match="No datasets provided."):
        _load_datasets(conf)


class TestInitModelAndOptimizer:
    """Test cases for the init model and optimizer function."""

    @pytest.fixture()
    def sdg_instance(self) -> MagicMock:
        """Get a mock instance for the sdg optimizer."""
        return MagicMock(autospec=torch.optim.SGD)

    @pytest.fixture()
    def sgd_mock(self, sdg_instance) -> MagicMock:
        """Mock the SGD optimizer"""
        with patch("torch.optim.SGD") as sdg:
            sdg.return_value = sdg_instance
            yield sdg

    @pytest.fixture()
    def model_instance(self) -> MagicMock:
        """Get a mock instance of the comb model."""
        return MagicMock(autospec=CombModel)

    @pytest.fixture()
    def model_mock(self, model_instance) -> MagicMock:
        """Mock the CombModel."""
        with patch("development.trainer.configured_training.CombModel") as model_mock:
            model_mock.return_value = model_instance
            yield model_mock

    @pytest.fixture()
    def pretrain_init(self, model_instance) -> MagicMock:
        """Mock the 'initialize from pretraining' function."""
        with patch(
            "development.trainer.configured_training.initialize_comb_model_from_pretraining"
        ) as init_mock:
            init_mock.return_value = model_instance
            yield init_mock

    @pytest.fixture()
    def ada_belief_instance(self) -> MagicMock:
        """Get a mock instance for the AdaBelief optimizer."""
        return MagicMock(autospec=AdaBelief)

    @pytest.fixture()
    def ada_belief_mock(self, ada_belief_instance) -> MagicMock:
        """Mock the AdaBelief optimizer."""
        with patch(
            "development.trainer.configured_training.AdaBelief"
        ) as ada_belief_mock:
            ada_belief_mock.return_value = ada_belief_instance
            yield ada_belief_mock

    def test_init_model_and_optimizer(
        self,
        model_mock: MagicMock,
        model_instance: MagicMock,
        sdg_instance: MagicMock,
        pretrain_init: MagicMock,
        sgd_mock: MagicMock,
    ):
        """Test that the model and optimizer can be initialized."""
        conf = TrainingConfig(
            datasets=["1", "2"],
            learning_rate=2e-4,
            pretraining_checkpoint="test_checkpoint",
        )

        model, optimizer = _init_model_and_optimizer(conf)

        assert model is model_instance
        assert optimizer is sdg_instance
        sgd_mock.assert_called_with(model_instance.parameters(), lr=2e-4, momentum=0.9)
        pretrain_init.assert_called_with(Path("test_checkpoint"), num_persons=2)

    def test_without_checkpoint(
        self,
        model_instance: MagicMock,
        model_mock: MagicMock,
        sdg_instance: MagicMock,
        pretrain_init: MagicMock,
        sgd_mock: MagicMock,
    ):
        """Test that the model and optimizer can be initialized."""
        conf = TrainingConfig(
            datasets=["1", "2"],
            learning_rate=2e-4,
        )

        model, optimizer = _init_model_and_optimizer(conf)

        assert model is model_instance
        assert optimizer is sdg_instance
        model_mock.assert_called_with(persons=2)
        sgd_mock.assert_called_with(model_instance.parameters(), lr=2e-4, momentum=0.9)
        pretrain_init.assert_not_called()

    def test_with_ada_belief(
        self,
        model_mock: MagicMock,
        sgd_mock: MagicMock,
        ada_belief_instance: MagicMock,
        ada_belief_mock: MagicMock,
    ):
        """Test that the AdaBelief optimizer can be initialized."""
        conf = TrainingConfig(
            datasets=["1", "2"], learning_rate=2e-5, optimizer="AdaBelief"
        )

        _, optimizer = _init_model_and_optimizer(conf)

        assert optimizer is ada_belief_instance
        sgd_mock.assert_not_called()
        ada_belief_mock.assert_called_once()
        assert ada_belief_mock.call_args.kwargs["lr"] == 2e-5

    def test_wrong_optimizer(
        self, model_mock: MagicMock, sgd_mock: MagicMock, ada_belief_mock: MagicMock
    ):
        """Test that a value error is raised if the wrong optimizer class is used."""
        wrong_conf = TrainingConfig(datasets=["1", "2"], optimizer="WrongOptimizer")

        with pytest.raises(
            ValueError,
            match="Optimizer wrong: 'WrongOptimizer'. Possible optimizer: 'SGD', 'AdaBelief'",
        ):
            _init_model_and_optimizer(wrong_conf)