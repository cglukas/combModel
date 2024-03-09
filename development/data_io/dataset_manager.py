"""Module for dataset management while training."""
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from development.data_io.dataloader2 import ImageSize, PersonDataset


class DatasetManager:
    """The dataset manager groups datasets and allows for easy access of batches."""

    def __init__(self, datasets: list[PersonDataset], shuffle: bool = True):
        """Initialize the dataset manager with the datasets.

        Args:
            datasets: list of datasets that should be managed.
            shuffle: pick data samples randomly. Default is True.
        """
        self._datasets = datasets
        self._batch_size = 8
        self._shuffle = shuffle

        self._dataloaders = []
        self._level = 0

        self.epoch_percent: float = 0.0
        """Percentage of the processed batches."""

    def set_batch_size(self, size: int):
        """Set the batch size.

        Args:
            size: amount of images per batch.
        """
        self._batch_size = size

    def set_level(self, level: int) -> None:
        """Set the current level of training to load appropriate image sizes.

        Args:
            level: image level between 0 and 8.
        """
        if not 0 <= level <= 8 or not isinstance(level, int):
            msg = "Only integer values in range between 0-8 is supported."
            raise ValueError(msg)
        for dataset in self._datasets:
            dataset.set_scale(ImageSize.from_index(level))
        self._start_dataloaders()

    def _start_dataloaders(self):
        """Initialize the dataloaders."""
        self._dataloaders = []
        for dataset in self._datasets:
            loader = DataLoader(
                dataset, batch_size=self._batch_size, shuffle=self._shuffle
            )
            self._dataloaders.append(loader)

    def iter_batches(self) -> list[torch.Tensor]:
        """Iterate over the datasets and for each a batch of samples.

        Notes:
            Smaller datasets will be repeated.

        Yields:
            A batch for each dataset.
        """
        if not self._dataloaders:
            self._start_dataloaders()

        max_length = max({len(dataset) for dataset in self._datasets})
        iterators = [iter(dataloader) for dataloader in self._dataloaders]
        max_batches = int(max_length / self._batch_size)
        for batch_num in range(max_batches):
            result = self._get_next_batch(iterators)

            self.epoch_percent = batch_num / max_batches
            yield result

    def _get_next_batch(
        self, iterators: list[Iterator[DataLoader]]
    ) -> list[torch.Tensor]:
        """Get the next batches from the dataloaders.

        If one dataloader is finished but, it will be instantiated as a new iterator.

        Args:
            iterators: list of dataloader iterators.

        Returns:
            list of batches.
        """
        result = []
        for loader_id, loader in enumerate(iterators):
            sample = next(loader, None)
            if sample is None:
                loader = iter(self._dataloaders[loader_id])
                iterators[loader_id] = loader
                sample = next(loader)
            result.append(sample)
        return result
