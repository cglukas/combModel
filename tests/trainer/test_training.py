from typing import Tuple

import pytest
from torch.utils.data import DataLoader

from code.trainer.training import Trainer


@pytest.mark.parametrize("sizes", [(1,3), (1,1), (10, 10), (8,1,3)])
def test_get_next_samples(sizes: Tuple):
    """Test if smaller datasets are repeated until the longest one is finished."""
    trainer = Trainer()
    trainer.dataloaders = [DataLoader([size]*size, batch_size=1) for size in sizes]
    longest_dataset = max(sizes)
    trainer._max_dataset_length = longest_dataset
    i = 0
    for sample in trainer.get_next_samples():
        assert all(sample)
        i += 1
    assert i == longest_dataset
