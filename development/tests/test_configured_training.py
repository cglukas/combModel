"""Tests for the training configuration."""

import pytest

from development.trainer.run_training import (
    run_training_for_single_config,
)
from development.trainer.configured_training.load_from_config import ConfigError
from development.trainer.configured_training.configuration import TrainingConfig


def test_run_training_wrong_config():
    """Test that resume and pretrain checkpoint together will raise an error."""
    conf = TrainingConfig(resume_checkpoint="test", pretraining_checkpoint="test")
    with pytest.raises(
        ConfigError,
        match="Resuming with pretrained checkpoint does not work. Only provide one value.",
    ):
        run_training_for_single_config(conf)
