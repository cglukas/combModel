"""Module for changing the level and blend of the model during training."""
from abc import ABC, abstractmethod


class AbstractLevelManager(ABC):
    """Base class for the level management."""

    def __init__(self):
        self.blend = 0
        self.level = 0

    @abstractmethod
    def increase_level_and_blend(self, score: float = 1.0) -> None:
        """Get the current level.

        Args:
            score: (optional) score between 0 (bad) and 1 (good).
        """


class LinearManager(AbstractLevelManager):
    """Basic level manager that will increase the level based on a constant blend rate."""

    def __init__(self, rate: float, max_level: int = 8):
        """Initialize the LinearManager.

        Args:
            rate: rate at which the blend and level are increased.
            max_level: highest level that can be reached.
        """
        super().__init__()
        self._blend_rate = rate
        self._max_level = max_level

    def increase_level_and_blend(self, score: float = 1.0) -> None:
        """Increase the level and blend.

        Args:
            score: this won't be used here.
        """
        self.blend += self._blend_rate
        if self.blend >= 1:
            self.blend = 0
            self.level = min(self._max_level, self.level + 1)


class ScoreGatedManager(AbstractLevelManager):
    """Level manager that blends with constant rate but only increases levels if a certain score is reached."""

    def __init__(self, rate: float, min_score: float, max_level: int = 8):
        """Initialize the manager.

        Args:
            rate: constant blend rate for increasing the level and blend.
            min_score: minimum score that needs to be reached for increasing the blend.
            max_level: highest level that can be reached.
        """
        super().__init__()
        self._blend_rate = rate
        self._min_score = min_score
        self._max_level = max_level

    def increase_level_and_blend(self, score: float = 1.0) -> None:
        """Increase the next level and blend if the score is high enough.

        Args:
            score: if the score is below the threshold, the level and blend are not increased.

        Returns:
            Next level and blend values.
        """
        if score < self._min_score:
            return

        self.blend += self._blend_rate
        if self.blend >= 1:
            self.blend = 0
            self.level = min(self._max_level, self.level + 1)


class ScoreGatedLevelManager(AbstractLevelManager):
    """A special manager that blends with a constant rate and only increases levels if the score is high enough."""

    def __init__(
        self,
        rate: float,
        min_score: float,
        max_level: int = 8,
        max_repeat: int | None = None,
    ):
        """Initialize the manager.

        Args:
            rate: constant blend rate used in between levels.
            min_score: minimum score that needs to be reached before the level is increased.
            max_level: highest level that can be reached.
            max_repeat: (optional) amount of repetitions until the level will be increased anyway.
        """
        super().__init__()
        self._repeat = 0
        self._blend_rate = rate
        self._min_score = min_score
        self._max_level = max_level
        self._max_repeat = max_repeat

    def increase_level_and_blend(self, score: float = 1.0):
        """Increase the level and blend.

        If the next level should start and the score is higher than min_score,
        the level will be increased. Else the old values will be used and the
        blend will be limited to 1.

        Args:
            score: current score of the model.
        """
        self.blend += self._blend_rate
        self.blend = min(self.blend, 1)
        if self.blend < 1:
            return

        if score < self._min_score:
            if self._max_repeat is None:
                # Only consider repetitions when max_repeat is set
                return

            self._repeat += 1
            if self._repeat <= self._max_repeat:
                return

        self._repeat = 0
        self.blend = 0
        self.level = min(self._max_level, self.level + 1)
