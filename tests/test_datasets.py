import pytest
from pytest import mark, raises
from os.path import join
import numpy as np
from torchmdnet.datasets import Custom


@mark.parametrize("energy", [True, False])
@mark.parametrize("forces", [True, False])
@mark.parametrize("num_files", [1, 3])
def test_custom(energy, forces, num_files, tmpdir, num_samples=100):
    # set up necessary files
    for i in range(num_files):
        np.save(
            join(tmpdir, f"coords_{i}.npy"), np.random.normal(size=(num_samples, 5, 3))
        )
        np.save(join(tmpdir, f"embed_{i}.npy"), np.random.randint(0, 100, size=(5)))
        if energy:
            np.save(
                join(tmpdir, f"energy_{i}.npy"),
                np.random.uniform(size=(num_samples, 1)),
            )
        if forces:
            np.save(
                join(tmpdir, f"forces_{i}.npy"),
                np.random.normal(size=(num_samples, 5, 3)),
            )

    # load data and test Custom dataset
    if energy == False and forces == False:
        with raises(AssertionError):
            Custom(
                coordglob=join(tmpdir, "coords*"), embedglob=join(tmpdir, "embed*"),
            )
        return

    data = Custom(
        coordglob=join(tmpdir, "coords*"),
        embedglob=join(tmpdir, "embed*"),
        energyglob=join(tmpdir, "energy*") if energy else None,
        forceglob=join(tmpdir, "forces*") if forces else None,
    )

    assert len(data) == num_samples * num_files, "Number of samples does not match"
    sample = data[0]
    assert hasattr(sample, "pos"), "Sample doesn't contain coords"
    assert hasattr(sample, "z"), "Sample doesn't contain atom numbers"
    if energy:
        assert hasattr(sample, "y"), "Sample doesn't contain energy"
    if forces:
        assert hasattr(sample, "dy"), "Sample doesn't contain forces"
