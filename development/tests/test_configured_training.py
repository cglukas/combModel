"""Tests for the training configuration."""
from unittest.mock import ANY, MagicMock, call, patch

import pytest
import torch

from development.data_io.dataset_manager import DatasetManager
from development.trainer.configured_training import TrainingConfig, _load_datasets


@pytest.mark.parametrize("datasets", [["test", "test2"], [], ["test"]])
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
