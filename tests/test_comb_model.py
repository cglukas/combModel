import pytest
import torch
from torch import nn

from development.model.comb_model import CombModel, Decoder, Encoder, torchscript_index_access


@pytest.mark.parametrize(
    ("num_modules", "index"),
    [
        (1, 0),
        (2, 0),
        (4, 2),
    ],
)
def test_torchscript_index_access(num_modules: int, index: int):
    """Test that the function replicates index accesses."""
    modules = [nn.ReLU() for _ in range(num_modules)]
    modules_list = nn.ModuleList(modules)
    parameter_list = nn.ParameterList(modules)

    found_module = torchscript_index_access(index=index, modules=modules_list)

    assert found_module == modules[index]

    found_module = torchscript_index_access(index=index, modules=parameter_list)
    assert found_module == modules[index]


def test_torchscript_index_access_index_out_of_range():
    """Test that an index error is raised."""
    with pytest.raises(IndexError, match="Index not found in list."):
        torchscript_index_access(index=1, modules=nn.ModuleList())


def test_comb_model():
    """Test that the expected tensor sizes will be processed correctly."""
    enc = Encoder()
    dec = Decoder()
    comb = CombModel()
    for level, size in enumerate([4, 8, 16, 32, 64, 128, 256, 512, 1024]):
        rand = torch.rand((1, 3, size, size))

        latent = enc.forward(rand, level=level)
        assert (1, 512, 1, 1) == latent.shape
        reconstruct = dec.forward(latent, level=level)
        assert rand.shape == reconstruct.shape

        reconstruct_comb = comb.progressive_forward(0, rand, level, 0.5)
        assert rand.shape == reconstruct_comb.shape
