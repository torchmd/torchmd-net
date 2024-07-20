import h5py
import psutil
import numpy as np
from pytest import mark
from os.path import join
from torchmdnet.datasets.mdcath import MDCATH
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def test_mdcath(tmpdir):
    num_atoms_list = np.linspace(50, 1000, 50)
    source_file = h5py.File(join(tmpdir, "mdcath_source.h5"), mode="w")
    for num_atoms in num_atoms_list:
        z = np.zeros(int(num_atoms))
        pos = np.zeros((100, int(num_atoms), 3))
        forces = np.zeros((100, int(num_atoms), 3))

        s_group = source_file.create_group(f"A{num_atoms}")

        s_group.attrs["numChains"] = 1
        s_group.attrs["numNoHAtoms"] = int(num_atoms) / 2
        s_group.attrs["numProteinAtoms"] = int(num_atoms)
        s_group.attrs["numResidues"] = int(num_atoms) / 10
        s_temp_group = s_group.create_group("348")
        s_replica_group = s_temp_group.create_group("0")
        s_replica_group.attrs["numFrames"] = 100
        s_replica_group.attrs["alpha"] = 0.30
        s_replica_group.attrs["beta"] = 0.25
        s_replica_group.attrs["coil"] = 0.45
        s_replica_group.attrs["max_gyration_radius"] = 2
        s_replica_group.attrs["max_num_neighbors_5A"] = 55
        s_replica_group.attrs["max_num_neighbors_9A"] = 200
        s_replica_group.attrs["min_gyration_radius"] = 1

        # write the dataset
        data = h5py.File(join(tmpdir, f"mdcath_dataset_A{num_atoms}.h5"), mode="w")
        group = data.create_group(f"A{num_atoms}")
        group.create_dataset("z", data=z)
        tempgroup = group.create_group("348")
        replicagroup = tempgroup.create_group("0")
        replicagroup.create_dataset("coords", data=pos)
        replicagroup.create_dataset("forces", data=forces)
        # add some attributes
        replicagroup.attrs["numFrames"] = 100
        replicagroup["coords"].attrs["unit"] = "Angstrom"
        replicagroup["forces"].attrs["unit"] = "kcal/mol/Angstrom"

        data.flush()
        data.close()

    dataset = MDCATH(root=tmpdir)
    dl = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    for _, data in enumerate(tqdm(dl)):
        pass


def test_mdcath_multiprocessing(tmpdir, num_entries=100, numFrames=10):
    # generate sample data
    z = np.zeros(num_entries)
    pos = np.zeros((numFrames, num_entries, 3))
    forces = np.zeros((numFrames, num_entries, 3))

    source_file = h5py.File(join(tmpdir, "mdcath_source.h5"), mode="w")
    s_group = source_file.create_group("A00")

    s_group.attrs["numChains"] = 1
    s_group.attrs["numNoHAtoms"] = num_entries / 2
    s_group.attrs["numProteinAtoms"] = num_entries
    s_group.attrs["numResidues"] = num_entries / 10
    s_temp_group = s_group.create_group("348")
    s_replica_group = s_temp_group.create_group("0")
    s_replica_group.attrs["numFrames"] = numFrames
    s_replica_group.attrs["alpha"] = 0.30
    s_replica_group.attrs["beta"] = 0.25
    s_replica_group.attrs["coil"] = 0.45
    s_replica_group.attrs["max_gyration_radius"] = 2
    s_replica_group.attrs["max_num_neighbors_5A"] = 55
    s_replica_group.attrs["max_num_neighbors_9A"] = 200
    s_replica_group.attrs["min_gyration_radius"] = 1

    # write the dataset
    data = h5py.File(join(tmpdir, "mdcath_dataset_A00.h5"), mode="w")
    group = data.create_group("A00")
    group.create_dataset("z", data=z)
    tempgroup = group.create_group("348")
    replicagroup = tempgroup.create_group("0")
    replicagroup.create_dataset("coords", data=pos)
    replicagroup.create_dataset("forces", data=forces)
    # add some attributes
    replicagroup.attrs["numFrames"] = numFrames
    replicagroup["coords"].attrs["unit"] = "Angstrom"
    replicagroup["forces"].attrs["unit"] = "kcal/mol/Angstrom"

    data.flush()
    data.close()

    # make sure creating the dataset doesn't open any files on the main process
    proc = psutil.Process()
    n_open = len(proc.open_files())

    dset = MDCATH(
        root=tmpdir,
    )
    assert len(proc.open_files()) == n_open, "creating the dataset object opened a file"


def replacer(arr, skipframes):
    tmp_arr = arr.copy()
    # function that take a numpy array of zeros and based on a skipframes value, replaces the zeros with 1s in that position
    for i in range(0, len(tmp_arr), skipframes):
        tmp_arr[i, :, :] = 1
    return tmp_arr


@mark.parametrize("skipframes", [1, 2, 5])
@mark.parametrize("batch_size", [1, 10])
@mark.parametrize("pdb_list", [["A50", "A612", "A1000"], None])
def test_mdcath_args(tmpdir, skipframes, batch_size, pdb_list):
    with h5py.File(join(tmpdir, "mdcath_source.h5"), mode="w") as source_file:
        num_frames_list = np.linspace(50, 1000, 50).astype(int)
        for num_frame in tqdm(num_frames_list, desc="Creating tmp files"):
            z = np.zeros(100)
            pos = np.zeros((num_frame, 100, 3))
            forces = np.zeros((num_frame, 100, 3))

            pos = replacer(pos, skipframes)
            forces = replacer(forces, skipframes)

            s_group = source_file.create_group(f"A{num_frame}")

            s_group.attrs["numChains"] = 1
            s_group.attrs["numNoHAtoms"] = 100 / 2
            s_group.attrs["numProteinAtoms"] = 100
            s_group.attrs["numResidues"] = 100 / 10
            s_temp_group = s_group.create_group("348")
            s_replica_group = s_temp_group.create_group("0")
            s_replica_group.attrs["numFrames"] = num_frame
            s_replica_group.attrs["alpha"] = 0.30
            s_replica_group.attrs["beta"] = 0.25
            s_replica_group.attrs["coil"] = 0.45
            s_replica_group.attrs["max_gyration_radius"] = 2
            s_replica_group.attrs["max_num_neighbors_5A"] = 55
            s_replica_group.attrs["max_num_neighbors_9A"] = 200
            s_replica_group.attrs["min_gyration_radius"] = 1

            # write the dataset
            data = h5py.File(join(tmpdir, f"mdcath_dataset_A{num_frame}.h5"), mode="w")
            group = data.create_group(f"A{num_frame}")
            group.create_dataset("z", data=z)
            tempgroup = group.create_group("348")
            replicagroup = tempgroup.create_group("0")
            replicagroup.create_dataset("coords", data=pos)
            replicagroup.create_dataset("forces", data=forces)
            # add some attributes
            replicagroup.attrs["numFrames"] = num_frame
            replicagroup["coords"].attrs["unit"] = "Angstrom"
            replicagroup["forces"].attrs["unit"] = "kcal/mol/Angstrom"

            data.flush()
            data.close()

    dataset = MDCATH(
        root=tmpdir, skipFrames=skipframes, pdb_list=pdb_list
    )
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    for _, data in enumerate(tqdm(dl)):
        # if the skipframes works correclty, data returned should be only 1s
        assert data.pos.all() == 1, "skipframes not working correctly for positions"
        assert data.neg_dy.all() == 1, "skipframes not working correctly for forces"
