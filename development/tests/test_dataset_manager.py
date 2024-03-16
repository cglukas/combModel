"""Tests for the dataset manager."""
from unittest.mock import MagicMock, patch

import pytest
import torch

from development.data_io.dataloader2 import ImageSize, PersonDataset
from development.data_io.dataset_manager import DatasetManager


def test_percent_epoch():
    """Test if the progress report works."""
    dataset = list(range(10))
    manager = DatasetManager([dataset])
    manager.set_batch_size(1)

    for i, _ in enumerate(manager.iter_batches()):
        progress = i / 10
        assert manager.epoch_percent == progress


@patch.object(DatasetManager, "_start_dataloaders")
def test_set_level(start_dataloaders: MagicMock):
    """Test that the level setting will be forwarded to the datasets."""
    person_set = MagicMock(spec=PersonDataset)
    manager = DatasetManager([person_set])

    manager.set_level(5)

    person_set.set_scale.assert_called_with(ImageSize.from_index(5))
    start_dataloaders.assert_called()


@pytest.mark.parametrize(("wrong_level"), [-1, 10, 1.54])
def test_set_level_wrong_level(wrong_level: float):
    """Test that an error is raised if the wrong level is set."""
    manager = DatasetManager([])
    with pytest.raises(
        ValueError, match="Only integer values in range between 0-8 is supported."
    ):
        manager.set_level(wrong_level)


@pytest.mark.parametrize("batchsize", [1, 2, 3])
def test_get_batches(batchsize: int):
    """Test that a batch can be loaded for each dataset."""
    dataset1 = [0, 1, 2]
    dataset2 = [0, 10, 20]
    manager = DatasetManager(datasets=[dataset1, dataset2], shuffle=False)
    manager.set_batch_size(batchsize)

    batches = next(iter(manager.iter_batches()))
    assert all(batches[0] == torch.tensor(dataset1[:batchsize]))
    assert all(batches[1] == torch.tensor(dataset2[:batchsize]))


@pytest.mark.parametrize("batchsize", [1, 2, 3])
def test_end_of_dataset(batchsize: int):
    """Test that datasets are reloaded when they reach the end."""
    small = [0, 1, 2]
    long = [0, 1, 2, 3, 4, 5]
    manager = DatasetManager(datasets=[small, long], shuffle=False)
    manager.set_batch_size(batchsize)

    for batches in manager.iter_batches():
        assert batches[0].nelement() != 0
        assert batches[1].nelement() != 0


@pytest.mark.parametrize("datasets", [[[0], [0]], [[0]], [], [[0]] * 5])
def test_num_datasets(datasets: list[list]):
    """Test that the amount of datasets can be read."""
    manager = DatasetManager(datasets=datasets)

    assert manager.num_datasets == len(datasets)
