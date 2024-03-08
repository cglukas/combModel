"""Module for dataset management while training."""
import torch


class DatasetManager:
    """The dataset manager groups datasets and allows for easy access of batches."""

    def set_batch_size(self, size: int):
        """Set the batch size.

        Args:
            size: amount of images per batch.
        """
    def set_level(self, level: int) -> None:
        """Set the current level of training to load appropriate image sizes.

        Args:
            level: image level between 0 and 8.
        """

    def get_batches(self) -> list[torch.Tensor]:
        """Get a batch for each dataset.

        Smaller datasets will be repeated.

        Returns:
            A batch for each dataset.
        """
