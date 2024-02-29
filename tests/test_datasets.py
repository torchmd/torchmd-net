# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import pytest
import os.path
from pytest import mark, raises
from os.path import join
import numpy as np
import psutil
from torchmdnet.datasets import Custom, HDF5, Ace
from torchmdnet.utils import write_as_hdf5
import h5py
import glob


def write_sample_npy_files(energy, forces, tmpdir, num_files):
    # set up necessary files
    n_atoms = np.random.randint(2, 10, size=num_files)
    num_samples = np.random.randint(10, 100, size=num_files)
    # n_atoms repeated num_samples times for each file
    for i in range(num_files):
        n_atoms_i = n_atoms[i]
        num_samples_i = num_samples[i]
        np.save(
            join(tmpdir, f"coords_{i}.npy"),
            np.random.normal(size=(num_samples_i, n_atoms_i, 3)).astype(np.float32),
        )
        np.save(
            join(tmpdir, f"embed_{i}.npy"), np.random.randint(0, 100, size=n_atoms_i)
        )
        if energy:
            np.save(
                join(tmpdir, f"energy_{i}.npy"),
                np.random.uniform(size=(num_samples_i, 1)).astype(np.float32),
            )
        if forces:
            np.save(
                join(tmpdir, f"forces_{i}.npy"),
                np.random.normal(size=(num_samples_i, n_atoms_i, 3)).astype(np.float32),
            )
    n_atoms_per_sample = []
    for i in range(num_files):
        n_atoms_per_sample.extend([n_atoms[i]] * num_samples[i])
    n_atoms_per_sample = np.array(n_atoms_per_sample)
    return n_atoms_per_sample


@mark.parametrize("energy", [True, False])
@mark.parametrize("forces", [True, False])
@mark.parametrize("num_files", [1, 3])
@mark.parametrize("preload", [True, False])
def test_custom(energy, forces, num_files, preload, tmpdir):
    # set up necessary files
    n_atoms_per_sample = write_sample_npy_files(energy, forces, tmpdir, num_files)

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

    assert len(data) == len(n_atoms_per_sample), "Number of samples does not match"
    sample = data[0]
    assert hasattr(sample, "pos"), "Sample doesn't contain coords"
    assert hasattr(sample, "z"), "Sample doesn't contain atom numbers"
    if energy:
        assert hasattr(sample, "y"), "Sample doesn't contain energy"
    if forces:
        assert hasattr(sample, "neg_dy"), "Sample doesn't contain forces"

    # Assert shapes of whole dataset:
    for i in range(len(data)):
        n_atoms_i = n_atoms_per_sample[i]
        assert np.array(data[i].z).shape == (
            n_atoms_i,
        ), "Dataset has incorrect atom numbers shape"
        assert np.array(data[i].pos).shape == (
            n_atoms_i,
            3,
        ), "Dataset has incorrect coords shape"
        if energy:
            assert np.array(data[i].y).shape == (
                1,
            ), "Dataset has incorrect energy shape"
        if forces:
            assert np.array(data[i].neg_dy).shape == (
                n_atoms_i,
                3,
            ), "Dataset has incorrect forces shape"
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


@mark.parametrize(("energy", "forces"), [(True, False), (False, True), (True, True)])
def test_write_as_hdf5(energy, forces, tmpdir):
    # set up necessary files
    num_files = 3
    write_sample_npy_files(energy, forces, tmpdir, num_files)
    files = {}
    files["pos"] = sorted(glob.glob(join(tmpdir, "coords*")))
    files["z"] = sorted(glob.glob(join(tmpdir, "embed*")))
    if energy:
        files["y"] = sorted(glob.glob(join(tmpdir, "energy*")))
    if forces:
        files["neg_dy"] = sorted(glob.glob(join(tmpdir, "forces*")))
    write_as_hdf5(files, join(tmpdir, "test.hdf5"))
    # Assert file is present in the disk
    assert os.path.isfile(join(tmpdir, "test.hdf5")), "HDF5 file was not created"
    # Assert shapes of whole dataset:
    data = h5py.File(join(tmpdir, "test.hdf5"), mode="r")
    assert len(data) == num_files
    for i in range(len(data)):
        pos_npy = np.load(files["pos"][i])
        n_samples = pos_npy.shape[0]
        n_atoms_i = pos_npy.shape[1]
        assert np.array(data[str(i)]["types"]).shape == (
            n_samples,
            n_atoms_i,
        ), "Dataset has incorrect atom numbers shape"
        assert np.array(data[str(i)]["pos"]).shape == (
            n_samples,
            n_atoms_i,
            3,
        ), "Dataset has incorrect coords shape"
        if energy:
            assert np.array(data[str(i)]["energy"]).shape == (
                n_samples,
                1,
            ), "Dataset has incorrect energy shape"
        if forces:
            assert np.array(data[str(i)]["forces"]).shape == (
                n_samples,
                n_atoms_i,
                3,
            ), "Dataset has incorrect forces shape"


@mark.parametrize("preload", [True, False])
@mark.parametrize(("energy", "forces"), [(True, False), (False, True), (True, True)])
@mark.parametrize("num_files", [1, 3])
def test_hdf5(preload, energy, forces, num_files, tmpdir):
    # set up necessary files
    n_atoms_per_sample = write_sample_npy_files(energy, forces, tmpdir, num_files)
    files = {}
    files["pos"] = sorted(glob.glob(join(tmpdir, "coords*")))
    files["z"] = sorted(glob.glob(join(tmpdir, "embed*")))
    if energy:
        files["y"] = sorted(glob.glob(join(tmpdir, "energy*")))
    if forces:
        files["neg_dy"] = sorted(glob.glob(join(tmpdir, "forces*")))
    write_as_hdf5(files, join(tmpdir, "test.hdf5"))
    # Assert file is present in the disk
    assert os.path.isfile(join(tmpdir, "test.hdf5")), "HDF5 file was not created"

    data = HDF5(join(tmpdir, "test.hdf5"), dataset_preload_limit=256 if preload else 0)

    assert len(data) == len(n_atoms_per_sample), "Number of samples does not match"
    sample = data[0]
    assert hasattr(sample, "pos"), "Sample doesn't contain coords"
    assert hasattr(sample, "z"), "Sample doesn't contain atom numbers"
    if energy:
        assert hasattr(sample, "y"), "Sample doesn't contain energy"
    if forces:
        assert hasattr(sample, "neg_dy"), "Sample doesn't contain forces"

    # Assert shapes of whole dataset:
    for i in range(len(data)):
        n_atoms_i = n_atoms_per_sample[i]
        assert np.array(data[i].z).shape == (
            n_atoms_i,
        ), "Dataset has incorrect atom numbers shape"
        assert np.array(data[i].pos).shape == (
            n_atoms_i,
            3,
        ), "Dataset has incorrect coords shape"
        if energy:
            assert np.array(data[i].y).shape == (
                1,
            ), "Dataset has incorrect energy shape"
        if forces:
            assert np.array(data[i].neg_dy).shape == (
                n_atoms_i,
                3,
            ), "Dataset has incorrect forces shape"
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


def test_ace(tmpdir):
    # Test Version 1.0
    tmpfilename = join(tmpdir, "molecule.h5")
    f = h5py.File(tmpfilename, "w")
    f.attrs["layout"] = "Ace"
    f.attrs["layout_version"] = "1.0"
    f.attrs["name"] = "sample_molecule_data"
    for m in range(3):  # Three molecules
        mol = f.create_group(f"mol_{m+1}")
        mol["atomic_numbers"] = [1, 6, 8]  # H, C, O
        mol["formal_charges"] = [0, 0, 0]  # Neutral charges
        confs = mol.create_group("conformations")
        for i in range(2):  # Two conformations
            conf = confs.create_group(f"conf_{i+1}")
            conf["positions"] = np.random.random((3, 3))
            conf["positions"].attrs["units"] = "Å"
            conf["formation_energy"] = np.random.random()
            conf["formation_energy"].attrs["units"] = "eV"
            conf["forces"] = np.random.random((3, 3))
            conf["forces"].attrs["units"] = "eV/Å"
            conf["partial_charges"] = np.random.random(3)
            conf["partial_charges"].attrs["units"] = "e"
            conf["dipole_moment"] = np.random.random(3)
            conf["dipole_moment"].attrs["units"] = "e*Å"

    dataset = Ace(root=tmpdir, paths=tmpfilename)
    assert len(dataset) == 6
    f.flush()
    f.close()
    # Test Version 2.0
    tmpfilename_v2 = join(tmpdir, "molecule_v2.h5")
    f2 = h5py.File(tmpfilename_v2, "w")
    f2.attrs["layout"] = "Ace"
    f2.attrs["layout_version"] = "2.0"
    f2.attrs["name"] = "sample_molecule_data_v2"
    master_mol_group = f2.create_group("master_molecule_group")
    for m in range(3):  # Three molecules
        mol = master_mol_group.create_group(f"mol_{m+1}")
        mol["atomic_numbers"] = [1, 6, 8]  # H, C, O
        mol["formal_charges"] = [0, 0, 0]  # Neutral charges
        mol["positions"] = np.random.random((2, 3, 3))  # Two conformations
        mol["positions"].attrs["units"] = "Å"
        mol["formation_energies"] = np.random.random(2)
        mol["formation_energies"].attrs["units"] = "eV"
        mol["forces"] = np.random.random((2, 3, 3))
        mol["forces"].attrs["units"] = "eV/Å"
        mol["partial_charges"] = np.random.random((2, 3))
        mol["partial_charges"].attrs["units"] = "e"
        mol["dipole_moment"] = np.random.random((2, 3))
        mol["dipole_moment"].attrs["units"] = "e*Å"
    dataset_v2 = Ace(root=tmpdir, paths=tmpfilename_v2)
    assert len(dataset_v2) == 6
    f2.flush()
    f2.close()
