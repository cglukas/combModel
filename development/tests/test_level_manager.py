"""Tests for the level managers."""
from development.trainer.level_manager import (
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


def test_linear_manager_max_level():
    """Test that the maximum level will never be exceeded."""
    manager = LinearManager(rate=100, max_level=2)
    manager.level = 2

    manager.increase_level_and_blend()

    assert manager.level == 2


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


