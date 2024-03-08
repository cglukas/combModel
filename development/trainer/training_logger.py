"""Module for logging the training."""
from datetime import datetime

import torch
import wandb


class TrainLogger:
    """Log progress on the terminal."""

    @staticmethod
    def log(level: int, blend: float, blend_rate: float, score: float, epoch: int):
        """Log a datapoint.

        Args:
            level: Current level of the training.
            blend: Current blend factor.
            blend_rate: Rate at which blending is happening.
            score: Current score of the network.
            epoch: Current epoch of teh training.
        """
        print(
            f"Time: {datetime.now().strftime('%H:%M:%S')}, Level {level} - {blend}, rate: {blend_rate}, score: {score}"
        )

    @staticmethod
    def log_image(image: torch.Tensor | wandb.Image, epoch: int):
        """Add an image to wandb.ai for the current epoch.

        Note:
            Don't spam this method as wandb will complain about large traffic.

        Args:
            image: Image to store.
            epoch: Current epoch that will be used for associating the image with
                   the model progress.
        """
        pass


class WandBLogger(TrainLogger):
    """Logger that reports additionally to wandb."""

    def __init__(
        self,
        project: str,
        entity: str,
        learning_rate: float,
        blend_rate: float,
        optimizer: str,
        continue_id: str = "",
    ):
        if not continue_id:
            wandb.init(project=project, entity=entity)
            wandb.config.update(
                {
                    "learning_rate": learning_rate,
                    "blend_rate": blend_rate,
                    "optimizer": optimizer,
                }
            )
        else:
            wandb.init(project=project, entity=entity, resume="must", id=continue_id)

    @staticmethod
    def log(level: int, blend: float, blend_rate: float, score: float, epoch: int):
        """Log a datapoint.

        Args:
            level: Current level of the training.
            blend: Current blend factor.
            blend_rate: Rate at which blending is happening.
            score: Current score of the network.
            epoch: Current epoch of teh training.
        """
        super().log(level, blend, blend_rate, score, epoch)
        wandb.log(
            {
                "score": score,
                "level": level + blend,
                "blend_rate": blend_rate,
                "epoch": epoch,
            }
        )

    @staticmethod
    def log_image(image: torch.Tensor | wandb.Image, epoch: int):
        """Add an image to wandb.ai for the current epoch.

        Note:
            Don't spam this method as wandb will complain about large traffic.

        Args:
            image: Image to store.
            epoch: Current epoch that will be used for associating the image with
                   the model progress.
        """
        wandb.log(
            {
                "epoch": epoch,
                "image": wandb.Image(image, caption="Reconstruction of model"),
            }
        )
