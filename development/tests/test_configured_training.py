"""Tests for the training configuration."""
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import pytest
import torch

from adabelief_pytorch import AdaBelief

from development.data_io.dataset_manager import DatasetManager
from development.model.comb_model import CombModel
from development.trainer.configured_training import (
    ConfigError,
    TrainingConfig,
    _init_model_and_optimizer,
    _load_datasets,
    _load_level_manager,
    _load_logger,
    _run_training,
    _yml_to_config,
)
from development.trainer.level_manager import ScoreGatedLevelManager
from development.trainer.training_logger import TrainLogger, WandBLogger


@pytest.mark.parametrize("datasets", [["test", "test2"], ["test"]])
@patch("development.trainer.configured_training.PersonDataset")
def test_load_datasets(dataset_mock: MagicMock, datasets: list[str]):
    """Test that the datasets are loaded into a dataset manager."""
    conf = TrainingConfig(datasets=datasets, device="cpu")
    device = torch.device("cpu")

    manager = _load_datasets(conf)

    assert isinstance(manager, DatasetManager)
    assert manager.num_datasets == len(datasets)
    assert dataset_mock.call_count == len(datasets)
    for dataset in datasets:
        dataset_mock.assert_any_call(dataset, device=device)


def test_load_datasets_empty() -> None:
    """Test that empty dataset lists will raise an error."""
    conf = TrainingConfig()

    with pytest.raises(ConfigError, match="No datasets provided."):
        _load_datasets(conf)


@pytest.mark.parametrize(
    "config_values",
    [
        {"rate": 0.05, "min_score": 0},
        {"rate": 0.05, "min_score": 0.9},
        {"rate": 0.05, "min_score": 0, "max_level": 4},
        {"rate": 0.1, "min_score": 0.5, "max_level": 6},
        {"rate": 0.05, "min_score": 0, "max_level": 4, "max_repeat": 8},
        {"rate": 0.05, "min_score": 0, "max_level": 4, "max_repeat": 12},
    ],
)
@patch(
    "development.trainer.configured_training.ScoreGatedLevelManager",
    autospec=ScoreGatedLevelManager,
)
def test_load_level_manager(manager: MagicMock, config_values: dict) -> None:
    """Test that the level manager will be initialized with the config values."""
    conf = TrainingConfig(level_manager_config=config_values)

    lvl_manager = _load_level_manager(conf)

    assert lvl_manager is manager.return_value
    manager.assert_called_with(**config_values)


@patch(
    "development.trainer.configured_training.ScoreGatedLevelManager",
    autospec=ScoreGatedLevelManager,
)
def test_load_level_manager_defaults(manager: MagicMock) -> None:
    """Test that the level manager will be initialized with the config values."""
    conf = TrainingConfig()

    _load_level_manager(conf)

    manager.assert_called_with(rate=0.05, min_score=0)


@patch(
    "development.trainer.configured_training.ScoreGatedLevelManager",
    autospec=ScoreGatedLevelManager,
)
def test_load_level_manager_wrong_config(manager: MagicMock) -> None:
    """Test if a value Error is raised if the config contains wrong values."""
    conf = TrainingConfig(level_manager_config={"wrong": 9})

    with pytest.raises(ConfigError, match="Wrong config values provided: 'wrong'."):
        _load_level_manager(conf)

    manager.assert_not_called()


def test_load_logger() -> None:
    """Test that a default logger is returned."""
    conf = TrainingConfig()

    assert isinstance(_load_logger(conf), TrainLogger)


@patch("development.trainer.configured_training.WandBLogger")
def test_load_logger_with_wandb_config(wandb_logger_mock: MagicMock) -> None:
    """Test if a wandb logger is loaded if the wandb config are provided."""
    conf = TrainingConfig(wandb={"project": "test", "user": "me"})
    logger_mock = MagicMock(autospec=WandBLogger)
    wandb_logger_mock.return_value = logger_mock

    logger = _load_logger(conf)

    assert logger is logger_mock
    assert wandb_logger_mock.call_args.kwargs["project"] == "test"
    assert wandb_logger_mock.call_args.kwargs["entity"] == "me"


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

    @pytest.mark.usefixtures("model_mock")
    def test_init_model_and_optimizer(
        self,
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

    @pytest.mark.usefixtures("model_mock", "pretrain_init")
    @patch("torch.device")
    def test_model_device(
        self, device_mock: MagicMock, model_instance: MagicMock, sgd_mock: MagicMock
    ):
        """Test that the model is on the right device before the optimizer loads its parameters."""
        conf = TrainingConfig(datasets=["1", "2"], device="cuda")
        call_order_manager = MagicMock()
        call_order_manager.attach_mock(model_instance.to, "to_device")
        call_order_manager.attach_mock(sgd_mock, "sgd_init")
        device_mock.return_value = "cuda_device"

        _init_model_and_optimizer(conf)

        assert call_order_manager.mock_calls == [
            call.to_device("cuda_device"),
            call.sgd_init(model_instance.parameters(), lr=ANY, momentum=ANY),
        ], "The optimizer was initialized before the model was on the right device."

    @pytest.mark.parametrize(("datasets"), [["0"], ["0"] * 3])
    def test_without_checkpoint(
        self,
        model_instance: MagicMock,
        model_mock: MagicMock,
        sdg_instance: MagicMock,
        pretrain_init: MagicMock,
        sgd_mock: MagicMock,
        datasets: list[list],
    ):
        """Test that the model and optimizer can be initialized."""
        conf = TrainingConfig(
            datasets=datasets,
            learning_rate=2e-4,
        )

        model, optimizer = _init_model_and_optimizer(conf)

        assert model is model_instance
        assert optimizer is sdg_instance
        model_mock.assert_called_with(persons=len(datasets))
        sgd_mock.assert_called_with(model_instance.parameters(), lr=2e-4, momentum=0.9)
        pretrain_init.assert_not_called()

    @pytest.mark.usefixtures("sgd_mock")
    def test_without_datasets(self, model_mock: MagicMock):
        """Test that a ConfigError is raised if no datasets are used."""
        conf = TrainingConfig()

        with pytest.raises(
            ConfigError,
            match="No datasets provided. Can't infer number of persons for the model.",
        ):
            _init_model_and_optimizer(conf)

        model_mock.assert_not_called()

    @pytest.mark.usefixtures("model_mock")
    def test_with_ada_belief(
        self,
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

    @pytest.mark.usefixtures("model_mock", "sgd_mock", "ada_belief_mock")
    def test_wrong_optimizer(self):
        """Test that a value error is raised if the wrong optimizer class is used."""
        wrong_conf = TrainingConfig(datasets=["1", "2"], optimizer="WrongOptimizer")

        with pytest.raises(
            ConfigError,
            match="Optimizer wrong: 'WrongOptimizer'. Possible optimizer: 'SGD', 'AdaBelief'",
        ):
            _init_model_and_optimizer(wrong_conf)


class TestYamlConfigParsing:
    """Tests for the yaml config parsing."""

    _EXPECTED_CONFIGS = [
        TrainingConfig(datasets=["1", "2"]),
        TrainingConfig(optimizer="AdaBelief"),
        TrainingConfig(wandb={"project": "test_project", "user": "me"}),
        TrainingConfig(
            device="gpu",
            learning_rate=1e-10,
            level_manager_config={"rate": 0.09, "min_score": 0.7},
        ),
        TrainingConfig(  # all possible fields used.
            name="TestTraining",
            optimizer="SGD",
            learning_rate=2e-5,
            trainings_folder="test_folder",
            pretraining_checkpoint="path/to/pretraining.pth",
            resume_checkpoint="path/to/resume_checkpoint.pth",  # This is technically wrong.
            device="GPU",
            datasets=["test1", "test2", "test3"],
            level_manager_config={
                "rate": 0.1,
                "min_score": 0.9,
                "max_level": 4,
                "max_repeat": 15,
            },
            wandb={"project": "test_project", "user": "me"},
        ),
    ]

    def setup_method(self):
        """Setup the testcase by loading the TrainingConfigs from the test yml."""
        with open(
            Path(__file__).parent / "test_configured_training_config.yml",
            encoding="utf-8",
        ) as file:
            self._loaded_configs = (  # pylint: disable=attribute-defined-outside-init
                _yml_to_config(file.read())
            )

    def test_configs_match(self):
        """Test that both configs match in size."""
        assert len(self._loaded_configs) == len(self._EXPECTED_CONFIGS)

    @pytest.mark.parametrize("index", range(len(_EXPECTED_CONFIGS)))
    def test_each_loaded_config(self, index: int):
        """Test each config one by one."""
        loaded_config = self._loaded_configs[index]
        assert loaded_config == self._EXPECTED_CONFIGS[index]


def test_run_training_wrong_config():
    """Test that resume and pretrain checkpoint together will raise an error."""
    conf = TrainingConfig(resume_checkpoint="test", pretraining_checkpoint="test")
    with pytest.raises(
        ConfigError,
        match="Resuming with pretrained checkpoint does not work. Only provide one value.",
    ):
        _run_training(conf)
