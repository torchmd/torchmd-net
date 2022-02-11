import torch
from torch.testing import assert_allclose
from pytest import mark
from glob import glob
from os.path import dirname, join
from torchmdnet.calculators import External
from torchmdnet.models.model import load_model

from utils import create_example_batch


def test_compare_forward():
    checkpoint = join(dirname(dirname(__file__)), "tests", "example.ckpt")
    z, pos, _ = create_example_batch(multiple_batches=False)
    calc = External(checkpoint, z.unsqueeze(0))
    model = load_model(checkpoint, derivative=True)

    e_calc, f_calc = calc.calculate(pos, None)
    e_pred, f_pred = model(z, pos)

    assert_allclose(e_calc, e_pred)
    assert_allclose(f_calc, f_pred.unsqueeze(0))


def test_compare_forward_multiple():
    checkpoint = join(dirname(dirname(__file__)), "tests", "example.ckpt")
    z1, pos1, _ = create_example_batch(multiple_batches=False)
    z2, pos2, _ = create_example_batch(multiple_batches=False)
    calc = External(checkpoint, torch.stack([z1, z2], dim=0))
    model = load_model(checkpoint, derivative=True)

    e_calc, f_calc = calc.calculate(torch.cat([pos1, pos2], dim=0), None)
    e_pred, f_pred = model(
        torch.cat([z1, z2]),
        torch.cat([pos1, pos2], dim=0),
        torch.cat([torch.zeros(len(z1)), torch.ones(len(z2))]).long(),
    )

    assert_allclose(e_calc, e_pred)
    assert_allclose(f_calc, f_pred.view(-1, len(z1), 3))
