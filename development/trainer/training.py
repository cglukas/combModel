"""Training utilities to run trainings with different configurations."""
from typing import List, Generator

import torch
from adabelief_pytorch import AdaBelief
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure

from development.model.comb_model import CombModel


class Trainer:
    """Training object to train a model."""

    def __init__(self):
        self.accumulated_score: float = 0.0
        """The accumulated score of all training steps in one epoch. To get a measure 
        of the model performance divide this with the samples of the epoch."""
        self.model = CombModel()
        """The model to train."""
        self.optimizer = None

        self.metric = StructuralSimilarityIndexMeasure()
        """The metric of the training needs to create a score. This means it needs to 
        increase if the output improves. It's also necessary that the metric can be used
        like `metric(predictions, targets)`"""
        self.dataloaders: List[DataLoader] = [DataLoader([1,2], batch_size=1), DataLoader([1,2,3,4], batch_size=1)]

        self._max_dataset_length = max(len(loader.dataset) for loader in self.dataloaders)
        """The length of the longest dataset. Every other dataset will be repeated in order
        to match this length."""

        self.current_person: int = 0
        """The current person that is trained. This will define which decoder of the
        model will be used and what dataloader will provide the data."""
        self.current_level: int = 0
        """The current level of the multilevel comb model. This defines which entry and
        exit layer of the model will be used for inference. It can be in the range of
        0-8."""
        self.current_blend: float = 0.0
        """The current influence of the next layer of the model. The allowed range is from
        0 to 1."""

    def train(self):
        """Start the training process."""

    def get_next_samples(self) -> Generator[List[torch.Tensor], None, None]:
        """Get the next samples of the dataloaders.

        This will iterate over all dataloaders and yield the samples of them
        until all samples of the longest dataset have been yielded once.
        Smaller datasets are repeated in this process.
        """
        batch_size = self.dataloaders[0].batch_size
        iterators = [iter(dataloader) for dataloader in self.dataloaders]
        for i in range(int(self._max_dataset_length/batch_size)):
            output = []

            for j, _iter in enumerate(iterators):
                try:
                    output.append(next(_iter))
                except StopIteration:
                    _iter = iter(self.dataloaders[j])
                    iterators[j] = _iter
                    # Reinitialize smaller dataloaders
                    output.append(next(_iter))
            yield output

    def train_one_batch(self, batch: torch.Tensor):
        """Train one batch and perform backpropagation on it.

        This will process the batch with the model. The fields `current_person`,
        `current_level` and `current_blend` influences this inference step.
        After the inference step, the metric will be used to calculate the difference
        between batch and inferred output. This will be used by the optimizer to
        perform one backward step on the model.

        Args:
            batch: current batch of data (shape [batchsize, channels, width, height])
        """
        inferred = self.model.progressive_forward(
            person=self.current_person,
            tensor=batch,
            level=self.current_level,
            last_level_influence=self.current_blend,
        )
        score: torch.Tensor = -self.metric(inferred, batch)
        score.backward()
        self.optimizer.step()
        self.accumulated_score = score.item()

if __name__ == '__main__':
    Trainer().train()