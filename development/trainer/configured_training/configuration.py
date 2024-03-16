"""Module for loading training configuration from files."""
from dataclasses import dataclass, field

import yaml


@dataclass
class TrainingConfig:
    """Dataclass for the training configuration."""

    name: str = ""
    """Optional name for the training configuration."""
    optimizer: str = "SGD"
    """Optimizer class to use. Possible values: 'SGD', 'AdaBelief'. Will not be used when training is resumed."""
    learning_rate: float = 10e-4
    """The learning rate for the optimizer."""
    trainings_folder: str = ""
    """Folder where separate trainings are saved."""
    pretraining_checkpoint: str = ""
    """Checkpoint of the pretraining. This can't be used in combination with the resume checkpoint."""
    resume_checkpoint: str = ""
    """The model checkpoint used for resuming. This can't be used in combination with the pretraining_checkpoint"""
    device: str = "cpu"
    """The device where the training should run on."""
    datasets: list[str] = field(default_factory=list)
    """The list of datasets that should be considered for training."""
    level_manager_config: dict[str, float] = field(default_factory=dict)
    """Configuration for the level management. See the documentation for the ScoreGatedLevelManager."""
    wandb: dict[str, str] = field(default_factory=dict)
    """Configuration for tracking the training with weights and biases. (see wandb.ai)"""


def yml_to_config(yml_text: str) -> list[TrainingConfig]:
    """Convert a yaml style text to the training config.

    Args:
        yml_text: yaml style configuration values.

    Returns:
        Initialized configuration.
    """
    all_configs = []
    for config in yaml.safe_load_all(yml_text):
        all_configs.append(TrainingConfig(**config))
    return all_configs
