"""Training utilities to run trainings with different configurations."""
from datetime import datetime
from pathlib import Path
from typing import List, Generator

import torch
import torchvision
from cv2 import cv2
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure

from development.data_io import dataloader
from development.model.comb_model import CombModel


class Trainer:
    """Training object to train a model."""

    def __init__(
        self,
        model: CombModel,
        optimizer,
        dataloaders: List[DataLoader],
        show_preview=False,
        device="cpu",
        save_epochs=5,
        max_level=8,
    ):

        self.max_level = min(max_level, 8)
        self.dataloaders: List[DataLoader] = dataloaders
        self.show_preview = show_preview
        self.device = device
        self.save_epochs = save_epochs
        self.train_start = datetime.now().strftime("%d-%m-%y_%H_%M")

        self.blend_rate = 1e-4
        """Rate at which the next level of the model is blended into the training."""
        self.accumulated_score: float = 0.0
        """The accumulated score of all training steps in one epoch. To get a measure 
        of the model performance divide this with the samples of the epoch."""
        self.model = model
        """The model to train."""
        self.model.to(device)
        self.optimizer: torch.optim.optimizer.Optimizer = optimizer

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
        self.current_level: int = 0
        """The current level of the multilevel comb model. This defines which entry and
        exit layer of the model will be used for inference. It can be in the range of
        0-8."""
        self.current_blend: float = 0.0
        """The current influence of the next layer of the model. The allowed range is from
        0 to 1."""
        self.epoch = 0

        self.training = True
        self.visualizer = TrainVisualizer()

    def train(self):
        """Start the training process."""
        while self.training:
            self.epoch += 1
            self.train_one_epoch()
            if self.epoch % self.save_epochs == 1:
                self.save()
            self._increase_blend_and_level()

    def save(self):
        filepath = (
            Path(r"C:\Users\Lukas\PycharmProjects\combModel\trainings")
            / f"{self.train_start}"
        )
        if not filepath.exists():
            print(f"{filepath} created")
            filepath.mkdir(exist_ok=True)

        torch.save(
            self.model.state_dict(),
            filepath
            / f"comb_model_{self.current_level}-{round(self.current_blend, 1)}.pth",
        )
        torch.save(
            self.optimizer.state_dict(),
            filepath
            / f"comb_model_optim_{self.current_level}-{round(self.current_blend, 1)}.pth",
        )

    def _increase_blend_and_level(self):
        """Increase the `current_blend` with the blend rate.

        If the blend gets above 1.0 it will fall back to 0. The `current_level`
        gets increased when the blend rate reached 1.0.
        """
        self.current_blend += self.blend_rate
        if self.current_blend > 1:
            self.current_blend = 0
            self.current_level += 1
        if self.current_level > self.max_level:
            # Stop the training.
            self.training = False

        self.current_level = min(self.current_level, self.max_level)

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
                self.current_person = (person + 1) % 2
                self.visualizer.add_image(img)
                swapped = self.process_batch(img.unsqueeze(dim=0))
                self.visualizer.add_image(swapped.squeeze())
        self.visualizer.show()

        epoch_score = self.accumulated_score / i
        print(
            f"Time: {datetime.now().strftime('%H:%M:%S')}, Level {self.current_level} - {self.current_blend}, rate: {self.blend_rate}, score: {epoch_score}"
        )

    def get_next_samples(self) -> Generator[List[tuple[torch.Tensor, torch.Tensor]], None, None]:
        """Get the next samples of the dataloaders.

        This will iterate over all dataloaders and yield the samples of them
        until all samples of the longest dataset have been yielded once.
        Smaller datasets are repeated in this process.
        """
        dataloader.SizeLoader.scale = dataloader.SCALES[self.current_level]
        batch_size = self.dataloaders[0].batch_size
        iterators = [iter(loader) for loader in self.dataloaders]
        for i in range(int(self._max_dataset_length / batch_size)):
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
        score: torch.Tensor = -self.metric(inferred, batch)
        score.backward()
        self.optimizer.step()
        self.accumulated_score += -score.item()

    def process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Process the batch with the current state of blend, person and level."""
        return self.model.progressive_forward(
            person=self.current_person,
            batch=batch,
            level=self.current_level,
            last_level_influence=self.current_blend,
        )


class TrainVisualizer:
    def __init__(self):
        self.previews: List[torch.Tensor] = []
        self.image = None

    def add_image(self, image: torch.Tensor):
        self.previews.append(image)

    def add_batches(self, source: torch.Tensor, processed: torch.Tensor):
        for src, prev in zip(source, processed):
            self.previews.append(src)
            self.previews.append(prev)

    def show(self):
        self.image = torchvision.utils.make_grid(self.previews, nrow=2)
        self.image = self.image.permute(1, 2, 0).detach().cpu().numpy()
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Deepfake Preview", self.image)
        cv2.waitKey(200)

    def clear(self):
        self.previews = []
