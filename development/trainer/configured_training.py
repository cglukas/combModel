"""File for training based on configuration files."""
from dataclasses import dataclass, field

import torch
from torch.optim import Optimizer

from development.data_io.dataloader2 import PersonDataset
from development.data_io.dataset_manager import DatasetManager
from development.model.comb_model import CombModel
from development.trainer.level_manager import AbstractLevelManager
from development.trainer.training_logger import TrainLogger


@dataclass
class TrainingConfig:
    device: str = "cpu"
    """The device where the training should run on."""
    datasets: list[str] = field(default_factory=list)
    """The list of datasets that should be considered for training."""


def _load_model(config: TrainingConfig) -> CombModel:
    pass


def _load_optimizer(config: TrainingConfig) -> Optimizer:
    pass


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
