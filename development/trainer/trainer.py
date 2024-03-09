"""Training utilities to run trainings with different configurations."""
import torch
from torch.optim import Optimizer
from torchmetrics import StructuralSimilarityIndexMeasure

from development.data_io.dataset_manager import DatasetManager
from development.model.comb_model import CombModel
from development.trainer.level_manager import AbstractLevelManager, LinearManager
from development.trainer.training_file_io import TrainingIO
from development.trainer.training_logger import TrainLogger
from development.trainer.visualizer import TrainVisualizer


class Trainer:
    """Training object to train a model."""

    def __init__(
        self,
        model: CombModel,
        optimizer: Optimizer,
        dataset_manager: DatasetManager,
        file_io: TrainingIO,
        device="cpu",
        save_epochs=5,
        logger: TrainLogger = None,
        level_manager: AbstractLevelManager | None = None,
    ):
        self.device = device
        self.save_epochs = save_epochs

        self.file_io = file_io
        self.level_manager = level_manager or LinearManager(rate=0.05)
        self.dataset_manager = dataset_manager

        self.accumulated_score: float = 0.0
        """The accumulated score of all training steps in one epoch. To get a measure 
        of the model performance divide this with the samples of the epoch."""
        self.model = model
        """The model to train."""
        self.model.to(device)

        self.optimizer: Optimizer = optimizer

        self.metric = StructuralSimilarityIndexMeasure()
        """The metric of the training needs to create a score. This means it needs to 
        increase if the output improves. It's also necessary that the metric can be used
        like `metric(predictions, targets)`"""
        self.metric.to(self.device)

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
                self.dataset_manager.set_level(self.level_manager.level)
        except Exception:
            raise
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
        self.logger.log_image(self.visualizer.image, self.epoch)
        self.file_io.save()

    def train_one_epoch(self):
        """Train the model for one epoch.

        One epoch contains all samples of all dataloaders.
        Smaller datasets are maybe repeated.
        """
        self.metric.reset()
        self.accumulated_score = 0
        i = 1  # prevent any zero division error
        last_images = {}
        for i, samples in enumerate(self.dataset_manager.iter_batches(), start=1):
            self.visualizer.clear()

            for person, single_sample in enumerate(samples):
                self.current_person = person
                test_img = single_sample[0][0]
                self.visualizer.add_image(test_img)
                self.train_one_batch(single_sample)
                last_images[person] = test_img

        with torch.no_grad():
            for person, img in last_images.items():
                self.current_person = (person + 1) % len(self.dataset_manager._datasets)
                self.visualizer.add_image(img)
                swapped = self.process_batch(img.unsqueeze(dim=0))
                self.visualizer.add_image(swapped.squeeze())
        self.visualizer.show()

        self.epoch_score = self.accumulated_score / i / len(self.dataset_manager._datasets)
        if self.logger:
            self.logger.log(
                level=self.level_manager.level,
                blend=self.level_manager.blend,
                blend_rate=0,
                score=self.epoch_score,
                epoch=self.epoch,
            )

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
