"""Tests for the level managers."""
import pytest

from development.trainer.level_manager import (
    EndOfLevelsReached,
    LinearManager,
    ScoreGatedManager,
    ScoreGatedLevelManager,
)


def test_linear_manager():
    """Test that the constant manager increases the level by a constant rate."""
    manger = LinearManager(rate=1)

    assert manger.level == 0
    assert manger.blend == 0

    manger.increase_level_and_blend()
    assert manger.level == 1
    assert manger.blend == 0

    manger.increase_level_and_blend()
    assert manger.level == 2
    assert manger.blend == 0


@pytest.mark.parametrize("max_level", [1, 4, 8])
def test_linear_manager_max_level(max_level: int):
    """Test that the maximum level will stop the training."""
    manager = LinearManager(rate=100, max_level=max_level)
    manager.level = max_level

    with pytest.raises(EndOfLevelsReached):
        manager.increase_level_and_blend()


def test_score_gated_manager():
    """Test if the level is only increased if the min_score is reached."""
    manager = ScoreGatedManager(rate=1, min_score=0.9)
    manager.increase_level_and_blend(score=0)
    assert manager.level == 0
    assert manager.blend == 0

    manager.increase_level_and_blend(score=0.9)
    assert manager.level == 1
    assert manager.blend == 0


def test_score_gated_level_manager():
    """Test if the level is only increased if the score is high enough."""
    manager = ScoreGatedLevelManager(rate=1, min_score=0.9)
    manager.level = 0
    manager.blend = 1

    manager.increase_level_and_blend(score=0)
    assert manager.level == 0
    assert manager.blend == 1

    manager.increase_level_and_blend(score=0.9)
    assert manager.level == 1
    assert manager.blend == 0


def test_score_gated_level_manager_max_repeat():
    """Test that the manager continues when the increase method got called x times."""
    manager = ScoreGatedLevelManager(rate=1, min_score=0.9, max_repeat=2)
    manager.level = 0
    manager.blend = 1

    # Two allowed repetitions:
    manager.increase_level_and_blend(score=0)
    manager.increase_level_and_blend(score=0)

    assert manager.level == 0
    assert manager.blend == 1

    # Third call:
    manager.increase_level_and_blend(score=0)

    assert manager.level == 1
    assert manager.blend == 0


def test_score_gated_level_manager_max_level():
    """Test that a EndOfLevelsReached exception is raised when the level is increased above the max_level."""
    manager = ScoreGatedLevelManager(rate=0.1, min_score=0, max_level=2)
    manager.level = 2
    manager.blend = 0.9

    with pytest.raises(EndOfLevelsReached):
        manager.increase_level_and_blend()
