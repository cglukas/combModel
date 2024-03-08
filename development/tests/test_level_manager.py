"""Tests for the level managers."""
from development.trainer.level_manager import LinearManager, ScoreGatedManager, ScoreGatedLevelManager


def test_linear_manager():
    """Test that the constant manager increases the level by a constant rate."""
    manger = LinearManager(rate=1)

    assert manger.get_next_level_and_blend() == (1, 0)
    assert manger.get_next_level_and_blend() == (2, 0)


def test_linear_manager_max_level():
    """Test that the maximum level will never be exceeded."""
    manager = LinearManager(rate=100, max_level=2)

    manager.level = 2

    assert manager.get_next_level_and_blend() == (2, 0)
    assert manager.get_next_level_and_blend() == (2, 0)


def test_score_gated_manager():
    """Test if the level is only increased if the min_score is reached."""
    manager = ScoreGatedManager(rate=1, min_score=0.9)
    assert manager.get_next_level_and_blend(score=0) == (0, 0)
    assert manager.get_next_level_and_blend(score=0) == (0, 0)
    assert manager.get_next_level_and_blend(score=0.9) == (1, 0)
    assert manager.get_next_level_and_blend(score=0.9) == (2, 0)


def test_score_gated_manager_start_with_score():
    """Test if the start is 0, 0."""
    manager = ScoreGatedManager(rate=1, min_score=0.9)
    assert manager.get_next_level_and_blend(score=0.9) == (1, 0)


def test_score_gated_level_manager():
    """Test if the level is only increased if the score is high enough."""
    manager = ScoreGatedLevelManager(rate=1, min_score=0.9)

    assert manager.get_next_level_and_blend(score=0) == (0, 1)
    assert manager.get_next_level_and_blend(score=0) == (0, 1)
    assert manager.get_next_level_and_blend(score=0.9) == (1, 0)

