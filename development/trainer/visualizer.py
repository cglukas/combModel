"""Module for the training visualizer."""
from typing import List

import cv2
import torch
import torchvision


class TrainVisualizer:
    """Visualizer for collecting and showing intermediate training results."""

    def __init__(self):
        self.previews: List[torch.Tensor] = []
        self.image = None

    def add_image(self, image: torch.Tensor):
        """Add a single image.

        Calling this method alternately with the source and processed image
        will result in a grid where on one side the source images are and
        on the other side the processed images.

        Args:
            image: image to display. Size of the image must be the same like
                   already added images. If other sized images neeed to be
                   displayed, call `clear` before that.
        """
        self.previews.append(image)

    def add_batches(self, source: torch.Tensor, processed: torch.Tensor):
        """Add a full batch of images.

        Source and processed images need to be of the same size and batch
        sizes need to be the same.

        Args:
            source: Batch of source images (will be on the left side of the grid).
            processed: Batch of processed images (will be on the right side of the grid).
        """
        for src, prev in zip(source, processed):
            self.previews.append(src)
            self.previews.append(prev)

    def show(self):
        """Display the previews in a new window."""
        self.image = torchvision.utils.make_grid(self.previews, nrow=2)
        self.image = self.image.permute(1, 2, 0).detach().cpu().numpy()
        cv2.imshow("Deepfake Preview", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(200)

    def clear(self):
        """Clear the loaded previews."""
        self.previews = []
