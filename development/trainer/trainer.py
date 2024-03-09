"""Training utilities to run trainings with different configurations."""
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure

from development.data_io import dataloader
from development.model.comb_model import CombModel
from development.trainer.level_manager import AbstractLevelManager, LinearManager
from development.trainer.training_logger import TrainLogger
from development.trainer.visualizer import TrainVisualizer


class Trainer:
    """Training object to train a model."""

    def __init__(
        self,
        model: CombModel,
        optimizer: Optimizer,
        dataloaders: List[DataLoader],
        device="cpu",
        save_epochs=5,
        logger: TrainLogger = None,
        level_manager: AbstractLevelManager | None = None,
    ):
        self.dataloaders: List[DataLoader] = dataloaders
        self.device = device
        self.save_epochs = save_epochs
        self.train_start = datetime.now().strftime("%d-%m-%y_%H_%M")
        self.level_manager = level_manager or LinearManager(rate=0.05)
        self.accumulated_score: float = 0.0
        """The accumulated score of all training steps in one epoch. To get a measure 
        of the model performance divide this with the samples of the epoch."""
        self.model = model
        """The model to train."""
        self.model.to(device)

        self.optimizer: Optimizer = optimizer
        self.optimizer.to(device)

        self.metric = StructuralSimilarityIndexMeasure()
        """The metric of the training needs to create a score. This means it needs to 
        increase if the output improves. It's also necessary that the metric can be used
        like `metric(predictions, targets)`"""
        self.metric.to(self.device)

        self._max_dataset_length = max(
            len(loader.dataset) for loader in self.dataloaders
        )
        """The length of the longest dataset. Every other dataset will be repeated in order
        to match this length."""

        self.current_person: int = 0
        """The current person that is trained. This will define which decoder of the
        model will be used and what dataloader will provide the data."""
        self.epoch = 0
        self.epoch_score = 0

        self.training = True
        self.visualizer = TrainVisualizer()
        self.logger = logger

    def train(self):
        """Start the training process."""
        try:
            while self.training:
                self.epoch += 1
                self.train_one_epoch()
                if self.epoch % self.save_epochs == 1:
                    self.save()
                # TODO: add stop condition if max level is reached.
                self.level_manager.increase_level_and_blend(score=self.epoch_score)
        finally:
            # Allways save the last training state.
            # TODO modify the file so that it can be differentiated from normal saves.
            self.save()

    def save(self):
        """Save the model and optimizer.

        This will save the model and optimizer state dicts to a file.
        The filename will contain information about level and blend
        for continuing the training.

        Additionally, this method logs the current image of the visualizer.
        """
        filepath = (
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\trainings")
            / f"{self.train_start}"
        )
        if not filepath.exists():
            print(f"{filepath} created")
            filepath.mkdir(exist_ok=True)
        self.logger.log_image(self.visualizer.image, self.epoch)
        # TODO [cglukas]: Model needs to be brought back to cpu before saving.
        torch.save(
            self.model.state_dict(),
            filepath
            / f"comb_model_{self.level_manager.level}-{round(self.level_manager.blend, 1)}.pth",
        )
        torch.save(
            self.optimizer.state_dict(),
            filepath
            / f"comb_model_optim_{self.level_manager.level}-{round(self.level_manager.blend, 1)}.pth",
        )

    def train_one_epoch(self):
        """Train the model for one epoch.

        One epoch contains all samples of all dataloaders.
        Smaller datasets are maybe repeated.
        """
        self.metric.reset()
        self.accumulated_score = 0
        i = 1  # prevent any zero division error
        last_images = {}
        for i, samples in enumerate(self.get_next_samples(), start=1):
            self.visualizer.clear()

            for person, single_sample in enumerate(samples):
                self.current_person = person
                test_img = single_sample[0][0]
                self.visualizer.add_image(test_img)
                self.train_one_batch(single_sample)
                last_images[person] = test_img

        with torch.no_grad():
            for person, img in last_images.items():
                self.current_person = (person + 1) % len(self.dataloaders)
                self.visualizer.add_image(img)
                swapped = self.process_batch(img.unsqueeze(dim=0))
                self.visualizer.add_image(swapped.squeeze())
        self.visualizer.show()

        self.epoch_score = self.accumulated_score / i / len(self.dataloaders)
        if self.logger:
            self.logger.log(
                level=self.level_manager.level,
                blend=self.level_manager.blend,
                blend_rate=0,
                score=self.epoch_score,
                epoch=self.epoch,
            )

    def get_next_samples(
        self,
    ) -> Iterator[List[tuple[torch.Tensor, torch.Tensor]]]:
        """Get the next samples of the dataloaders.

        This will iterate over all dataloaders and yield the samples of them
        until all samples of the longest dataset have been yielded once.
        Smaller datasets are repeated in this process.
        """
        dataloader.SizeLoader.scale = dataloader.SCALES[self.level_manager.level]
        batch_size = self.dataloaders[0].batch_size
        iterators = [iter(loader) for loader in self.dataloaders]
        for _ in range(int(self._max_dataset_length / batch_size)):
            output = []

            for j, _iter in enumerate(iterators):
                try:
                    sample = next(_iter)
                except StopIteration:
                    _iter = iter(self.dataloaders[j])
                    iterators[j] = _iter
                    # Reinitialize smaller dataloaders
                    sample = next(_iter)
                sample = sample[0].to(self.device), sample[1].to(self.device)
                output.append(sample)
            yield output

    def train_one_batch(self, batch: tuple[torch.Tensor, torch.Tensor]):
        """Train one batch and perform backpropagation on it.

        This will process the batch with the model. The fields `current_person`,
        `current_level` and `current_blend` influences this inference step.
        After the inference step, the metric will be used to calculate the difference
        between batch and inferred output. This will be used by the optimizer to
        perform one backward step on the model.

        Args:
            batch: batch of images and masks (tensor shapes [batchsize, channels, width, height])
        """
        self.optimizer.zero_grad()
        batch, mask = batch
        inferred = self.process_batch(batch)
        inferred = inferred * mask
        self.visualizer.add_image(inferred[0])
        batch = batch * mask
        if batch.shape[-1] < 16:
            batch = torch.nn.functional.interpolate(batch, (16, 16))
            inferred = torch.nn.functional.interpolate(inferred, (16, 16))
        score: torch.Tensor = -1 * self.metric(inferred, batch)
        score.backward()
        self.optimizer.step()
        self.accumulated_score += -score.item()

    def process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Process the batch with the current state of blend, person and level."""
        return self.model.progressive_forward(
            person=self.current_person,
            batch=batch,
            level=self.level_manager.level,
            last_level_influence=self.level_manager.blend,
        )
