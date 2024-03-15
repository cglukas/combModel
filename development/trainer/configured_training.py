"""File for training based on configuration files."""
from dataclasses import dataclass, field
from pathlib import Path

import torch
from adabelief_pytorch import AdaBelief
from torch.optim import Optimizer

from development.data_io.dataloader2 import PersonDataset
from development.data_io.dataset_manager import DatasetManager
from development.model.comb_model import CombModel
from development.model.utils import initialize_comb_model_from_pretraining
from development.trainer.level_manager import AbstractLevelManager
from development.trainer.training_logger import TrainLogger


@dataclass
class TrainingConfig:
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


def _resume_training(config: TrainingConfig) -> tuple[CombModel, Optimizer]:
    pass


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

    return model, optimizer


def _load_logger(config: TrainingConfig) -> TrainLogger:
    pass


def _load_level_manager(config: TrainingConfig) -> AbstractLevelManager:
    pass


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
