"""File for training based on configuration files."""
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from adabelief_pytorch import AdaBelief
from torch.optim import Optimizer

from development.data_io.dataloader2 import PersonDataset
from development.data_io.dataset_manager import DatasetManager
from development.model.comb_model import CombModel
from development.model.utils import initialize_comb_model_from_pretraining
from development.trainer.level_manager import (
    ScoreGatedLevelManager,
)
from development.trainer.trainer import Trainer
from development.trainer.training_file_io import TrainingIO
from development.trainer.training_logger import TrainLogger, WandBLogger


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


def _init_model_and_optimizer(config: TrainingConfig) -> tuple[CombModel, Optimizer]:
    """Initialize the model and optimizer based on the config.

    If a pretraining is specified, this copies the pretraining to all decoders of the model.
    """
    persons = len(config.datasets)

    if config.pretraining_checkpoint:
        checkpoint = Path(config.pretraining_checkpoint)
        model = initialize_comb_model_from_pretraining(checkpoint, num_persons=persons)
    else:
        model = CombModel(persons=persons)

    return model, _get_optimizer(config, model)


def _get_optimizer(config: TrainingConfig, model: CombModel) -> Optimizer:
    """Get the optimizer for the model.

    Args:
        config: config to drive optimizer class and learning rate.
        model: used for parameter assignment on the optimizer.

    Returns:
        Initialized optimizer.
    """
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, momentum=0.9
        )
    elif config.optimizer == "AdaBelief":
        # GAN(small) settings for the AdaBelief optimizer:
        # more configurations can be found here:
        # https://github.com/juntang-zhuang/Adabelief-Optimizer#hyper-parameters-in-pytorch
        optimizer = AdaBelief(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.5, 0.999),
            eps=1e-12,
            weight_decay=0,
            weight_decouple=True,
            rectify=False,
            fixed_decay=False,
        )
    else:
        msg = f"Optimizer wrong: '{config.optimizer}'. Possible optimizer: 'SGD', 'AdaBelief'."
        raise ValueError(msg)
    return optimizer


def _load_logger(config: TrainingConfig) -> TrainLogger:
    """Get a logger for the training."""
    if config.wandb:
        if "project" not in config.wandb or "user" not in config.wandb:
            raise ValueError("Can only log to W&B with user and project provided.")
        return WandBLogger(
            project=config.wandb["project"],
            entity=config.wandb["user"],
            learning_rate=config.learning_rate,
            blend_rate=0,
            optimizer=config.optimizer,
        )
    return TrainLogger()


def _load_level_manager(config: TrainingConfig) -> ScoreGatedLevelManager:
    """Initialize the level manager with the config values."""
    defaults = {"rate": 0.05, "min_score": 0}
    manager_config = config.level_manager_config

    allowed_args = inspect.getfullargspec(ScoreGatedLevelManager).args
    if "self" in allowed_args:
        allowed_args.remove("self")  # self is the class reference.

    argument_mismatch = set(manager_config.keys()).difference(set(allowed_args))
    if argument_mismatch:
        wrong_args = [f"'{arg}'" for arg in argument_mismatch]
        msg = f"Wrong config values provided: {','.join(wrong_args)}."
        raise ValueError(msg)

    defaults.update(manager_config)
    return ScoreGatedLevelManager(**defaults)


def _load_datasets(config: TrainingConfig) -> DatasetManager:
    """Load the datasets from the config."""
    if len(config.datasets) == 0:
        msg = "No datasets provided."
        raise ValueError(msg)
    datasets = []
    device = torch.device(config.device)
    for path in config.datasets:
        datasets = PersonDataset(path, device=device)
    return DatasetManager(datasets=datasets)


def _yml_to_config(yml_text: str) -> list[TrainingConfig]:
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
