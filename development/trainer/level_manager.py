"""Module for changing the level and blend of the model during training."""
from abc import ABC, abstractmethod


class AbstractLevelManager(ABC):
    """Base class for the level management."""

    def __init__(self):
        self.blend = 0
        self.level = 0

    def get_start_values(self) -> tuple[int, float]:
        """Get the start values of level and blend.

        Returns:
            Level, blend as a tuple.
        """
        return self.level, self.blend

    @abstractmethod
    def get_next_level_and_blend(self, score: float = 1.0) -> tuple[int, float]:
        """Get the current level.

        Args:
            score: (optional) score between 0 (bad) and 1 (good).

        Returns:
           level, blend as a tuple.
        """
        pass


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

    def get_next_level_and_blend(self, score: float = 1.0) -> tuple[int, float]:
        """Get the next level and blend.

        Args:
            score: this won't be used here.

        Returns:
            Level and blend value for the current time.
        """
        self.blend += self._blend_rate
        if self.blend >= 1:
            self.blend = 0
            self.level = min(self._max_level, self.level+1)

        return self.level, self.blend


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

    def get_next_level_and_blend(self, score: float = 1.0) -> tuple[int, float]:
        """Get the next level and blend.

        Args:
            score: if the score is below the threshold, the level and blend are not increased.

        Returns:
            Next level and blend values.
        """
        if score < self._min_score:
            return self.level, self.blend

        self.blend += self._blend_rate
        if self.blend >= 1:
            self.blend = 0
            self.level = min(self._max_level, self.level+1)

        return self.level, self.blend


class ScoreGatedLevelManager(AbstractLevelManager):
    """A special manager that blends with a constant rate and only increases levels if the score is high enough."""

    def __init__(self, rate: float, min_score: float, max_level: int = 8):
        """Initialize the manager.

        Args:
            rate: constant blend rate used in between levels.
            min_score: minimum score that needs to be reached before the level is increased.
            max_level: highest level that can be reached.
        """
        super().__init__()
        self._blend_rate = rate
        self._min_score = min_score
        self._max_level = max_level
    def get_next_level_and_blend(self, score: float = 1.0) -> tuple[int, float]:
        """Get the next level and blend.

        Args:
            score: current score of the model.

        Returns:
             If the score is higher than min_score, the level will be increased.
             Else the old values will be returned.
        """
        self.blend += self._blend_rate
        self.blend = min(self.blend, 1)
        if self.blend == 1 and score >= self._min_score:
            self.blend = 0
            self.level = min(self._max_level, self.level+1)

        return self.level, self.blend
