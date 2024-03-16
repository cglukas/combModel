"""Tests for the configuration reading."""
from pathlib import Path

import pytest

from development.trainer.configured_training.configuration import (
    TrainingConfig,
    yml_to_config,
)


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
            # This is technically wrong: resume_checkpoint and pretraining_checkpoint would never be used together.
            resume_checkpoint="path/to/resume_checkpoint.pth",
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
                yml_to_config(file.read())
            )

    def test_configs_match(self):
        """Test that both configs match in size."""
        assert len(self._loaded_configs) == len(self._EXPECTED_CONFIGS)

    @pytest.mark.parametrize("index", range(len(_EXPECTED_CONFIGS)))
    def test_each_loaded_config(self, index: int):
        """Test each config one by one."""
        loaded_config = self._loaded_configs[index]
        assert loaded_config == self._EXPECTED_CONFIGS[index]
