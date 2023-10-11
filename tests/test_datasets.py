import pytest
from pytest import mark, raises
from os.path import join
import numpy as np
import psutil
from torchmdnet.datasets import Custom, HDF5
import h5py


@mark.parametrize("energy", [True, False])
@mark.parametrize("forces", [True, False])
@mark.parametrize("num_files", [1, 3])
@mark.parametrize("preload", [False, True])
def test_custom(energy, forces, num_files, preload, tmpdir, num_samples=100):
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
                coordglob=join(tmpdir, "coords*"),
                embedglob=join(tmpdir, "embed*"),
            )
        return

    data = Custom(
        coordglob=join(tmpdir, "coords*"),
        embedglob=join(tmpdir, "embed*"),
        energyglob=join(tmpdir, "energy*") if energy else None,
        forceglob=join(tmpdir, "forces*") if forces else None,
        preload_memory_limit=256 if preload else 0,
    )

    assert len(data) == num_samples * num_files, "Number of samples does not match"
    sample = data[0]
    assert hasattr(sample, "pos"), "Sample doesn't contain coords"
    assert hasattr(sample, "z"), "Sample doesn't contain atom numbers"
    if energy:
        assert hasattr(sample, "y"), "Sample doesn't contain energy"
    if forces:
        assert hasattr(sample, "neg_dy"), "Sample doesn't contain forces"

    # Assert sample has the correct values

    # get the reference values from coords_0.npy and embed_0.npy
    ref_pos = np.load(join(tmpdir, "coords_0.npy"))[0]
    ref_z = np.load(join(tmpdir, "embed_0.npy"))
    assert np.allclose(sample.pos, ref_pos), "Sample has incorrect coords"
    assert np.allclose(sample.z, ref_z), "Sample has incorrect atom numbers"
    if energy:
        ref_y = np.load(join(tmpdir, "energy_0.npy"))[0] if energy else None
        assert np.allclose(sample.y, ref_y), "Sample has incorrect energy"
    if forces:
        ref_neg_dy = np.load(join(tmpdir, "forces_0.npy"))[0] if forces else None
        assert np.allclose(sample.neg_dy, ref_neg_dy), "Sample has incorrect forces"


def test_hdf5_multiprocessing(tmpdir, num_entries=100):
    # generate sample data
    z = np.zeros(num_entries)
    pos = np.zeros(num_entries * 3).reshape(-1, 3)
    energy = np.zeros(num_entries)

    # write the dataset
    data = h5py.File(join(tmpdir, "test_hdf5_multiprocessing.h5"), mode="w")
    group = data.create_group("group")
    group["types"] = z[:, None]
    group["pos"] = pos[:, None]
    group["energy"] = energy
    data.flush()
    data.close()

    # make sure creating the dataset doesn't open any files on the main process
    proc = psutil.Process()
    n_open = len(proc.open_files())

    dset = HDF5(join(tmpdir, "test_hdf5_multiprocessing.h5"))

    assert len(proc.open_files()) == n_open, "creating the dataset object opened a file"
