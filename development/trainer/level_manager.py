"""Module for changing the level and blend of the model during training."""
from abc import ABC, abstractmethod


class EndOfLevelsReached(Exception):
    """Exception if the level manager will increment above the max level."""


class AbstractLevelManager(ABC):
    """Base class for the level management."""

    def __init__(self):
        self._blend = 0
        self._level = 0
        self._max_level = 8
        self._level_bound_exception = ValueError("Only level in range 0-8 are allowed.")

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        if not isinstance(value, int):
            msg = "Only int is supported"
            raise ValueError(msg)
        if not (0 <= value <= 8):
            raise self._level_bound_exception
        self._level = value

    @property
    def blend(self):
        return self._blend

    @blend.setter
    def blend(self, value: float | int):
        if not isinstance(value, (float, int)):
            msg = "Only float or int values allowed."
            raise ValueError(msg)
        if not (0 <= value <= 1):
            msg = "Only values in range 0-1 are allowed."
            raise ValueError(msg)
        self._blend = value

    @property
    def max_level(self):
        return self._max_level

    @max_level.setter
    def max_level(self, value: int):
        if not isinstance(value, int):
            msg = "Only int is supported"
            raise ValueError(msg)
        if not (0 < value <= 8):
            msg = "Only int in range 1-8 are allowed."
            raise ValueError(msg)
        self._max_level = value

    @abstractmethod
    def _handle_level_and_blend(self, score: float = 1.0) -> None:
        """Process the level and blend handling.

        Args:
            score: value between 0-1

        Returns:

        """
        pass

    def increase_level_and_blend(self, score: float = 1.0) -> None:
        """Get the current level.

        Args:
            score: (optional) score between 0 (bad) and 1 (good).
        """
        if not (0 <= score <= 1):
            msg = "Only score in range 0-1 is allowed."
            raise ValueError(msg)

        try:
            self._handle_level_and_blend(score)
        except ValueError as e:
            if e == self._level_bound_exception:
                raise EndOfLevelsReached() from e
            raise e
        if self.level > self._max_level:
            raise EndOfLevelsReached()


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

    def _handle_level_and_blend(self, score: float = 1.0):
        """Increase the level and blend.

        Args:
            score: this won't be used here.
        """
        self.blend = min(self.blend + self._blend_rate, 1)
        if self.blend >= 1:
            self.blend = 0
            self.level += 1


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

    def _handle_level_and_blend(self, score: float = 1.0) -> None:
        """Increase the next level and blend if the score is high enough.

        Args:
            score: if the score is below the threshold, the level and blend are not increased.

        Returns:
            Next level and blend values.
        """
        if score < self._min_score:
            return

        self.blend = min(self.blend + self._blend_rate, 1)
        if self.blend >= 1:
            self.blend = 0
            self.level += 1


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

    def _handle_level_and_blend(self, score: float = 1.0):
        """Increase the level and blend.

        If the next level should start and the score is higher than min_score,
        the level will be increased. Else the old values will be used and the
        blend will be limited to 1.

        Args:
            score: current score of the model.
        """
        self.blend = min(self.blend + self._blend_rate, 1)
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
        self.level += 1
