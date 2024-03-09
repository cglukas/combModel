"""Module for saving and loading training relevant files."""
import re
from pathlib import Path

import torch
from torch.optim import Optimizer
from development.model.comb_model import CombModel
from development.trainer.level_manager import AbstractLevelManager


class TrainingIO:
    """Handle model and optimizer loading and saving."""

    LVL_BLEND_RE = re.compile(r".+(?P<level>\d)_(?P<blend>\d\.\d+)")

    def __init__(
        self,
        model: CombModel,
        optimizer: Optimizer,
        level_manager: AbstractLevelManager,
    ):
        """Initialize the IO handler.

        Args:
            model: model to load/ save.
            optimizer: optimizer to load/ save.
            level_manager: level manager to get and set level and blend information.
        """
        self._model = model
        self._optimizer = optimizer
        self._level_manager = level_manager
        self._folder: Path | None = None

    def set_folder(self, folder: Path):
        """Set the fallback folder for saving."""
        self._folder = folder

    def save(self, folder: Path | None = None):
        """Save the model and optimizer.

        Args:
            folder: folder where the files should be saved.
        """
        folder = folder or self._folder
        model_file = folder / f"model_{self._get_lvl_blend_string()}.pth"
        # TODO [cglukas]: Model needs to be brought back to cpu before saving.
        torch.save(self._model.state_dict(), str(model_file))

        optim_file = folder / f"optim_{self._get_lvl_blend_string()}.pth"
        torch.save(self._optimizer.state_dict(), str(optim_file))

    def load(self, model_filepath: Path):
        """Load the model and optimizer from the provided folder.

        This also stores the last level and blend values on the level_manager.

        Args:
            model_filepath: filepath to the saved model file.
        """
        match = self.LVL_BLEND_RE.match(model_filepath.name)
        if not match:
            msg = f"Can't extract level and blend values from filename: '{model_filepath.name}'"
            raise RuntimeError(msg)
        self._level_manager.level = int(match["level"])
        self._level_manager.blend = float(match["blend"])

        optim_filepath = (
            model_filepath.parent / f"optim_{self._get_lvl_blend_string()}.pth"
        )
        model_state = torch.load(str(model_filepath))
        optim_state = torch.load(str(optim_filepath))

        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(optim_state)

    def _get_lvl_blend_string(self) -> str:
        """Get the level and blend formatted for the filepaths."""
        return (
            f"{self._level_manager.level}_{round(float(self._level_manager.blend), 2)}"
        )
