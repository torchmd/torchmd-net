# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import yaml
import argparse
import numpy as np
import torch
from os.path import dirname, join, exists
from lightning_utilities.core.rank_zero import rank_zero_warn
import functools
import warnings

# fmt: off
# Atomic masses are based on:
#
#   Meija, J., Coplen, T., Berglund, M., et al. (2016). Atomic weights of
#   the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry,
#   88(3), pp. 265-291. Retrieved 30 Nov. 2016,
#   from doi:10.1515/pac-2015-0305
#
# Standard atomic weights are taken from Table 1: "Standard atomic weights
# 2013", with the uncertainties ignored.
# For hydrogen, helium, boron, carbon, nitrogen, oxygen, magnesium, silicon,
# sulfur, chlorine, bromine and thallium, where the weights are given as a
# range the "conventional" weights are taken from Table 3 and the ranges are
# given in the comments.
# The mass of the most stable isotope (in Table 4) is used for elements
# where there the element has no stable isotopes (to avoid NaNs): Tc, Pm,
# Po, At, Rn, Fr, Ra, Ac, everything after N
atomic_masses = np.array([
    1.0, 1.008, 4.002602, 6.94, 9.0121831,
    10.81, 12.011, 14.007, 15.999, 18.998403163,
    20.1797, 22.98976928, 24.305, 26.9815385, 28.085,
    30.973761998, 32.06, 35.45, 39.948, 39.0983,
    40.078, 44.955908, 47.867, 50.9415, 51.9961,
    54.938044, 55.845, 58.933194, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.921595, 78.971,
    79.904, 83.798, 85.4678, 87.62, 88.90584,
    91.224, 92.90637, 95.95, 97.90721, 101.07,
    102.9055, 106.42, 107.8682, 112.414, 114.818,
    118.71, 121.76, 127.6, 126.90447, 131.293,
    132.90545196, 137.327, 138.90547, 140.116, 140.90766,
    144.242, 144.91276, 150.36, 151.964, 157.25,
    158.92535, 162.5, 164.93033, 167.259, 168.93422,
    173.054, 174.9668, 178.49, 180.94788, 183.84,
    186.207, 190.23, 192.217, 195.084, 196.966569,
    200.592, 204.38, 207.2, 208.9804, 208.98243,
    209.98715, 222.01758, 223.01974, 226.02541, 227.02775,
    232.0377, 231.03588, 238.02891, 237.04817, 244.06421,
    243.06138, 247.07035, 247.07031, 251.07959, 252.083,
    257.09511, 258.09843, 259.101, 262.11, 267.122,
    268.126, 271.134, 270.133, 269.1338, 278.156,
    281.165, 281.166, 285.177, 286.182, 289.19,
    289.194, 293.204, 293.208, 294.214,
])
# fmt: on

ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}


def train_val_test_split(dset_len, train_size, val_size, test_size, seed, order=None):
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(
    dataset_len,
    train_size,
    val_size,
    test_size,
    seed,
    filename=None,
    splits=None,
    order=None,
):
    if splits is not None:
        splits = np.load(splits, allow_pickle=True)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            if (
                "load_model" in config
                and namespace.load_model is not None
                and config["load_model"] != namespace.load_model
            ):
                rank_zero_warn(
                    f"The load model argument was specified as a command line argument "
                    f"({namespace.load_model}) and in the config file ({config['load_model']}). "
                    f"Ignoring 'load_model' from the config file and loading {namespace.load_model}."
                )
                del config["load_model"]
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


class LoadFromCheckpoint(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        hparams_path = join(dirname(values), "hparams.yaml")
        if not exists(hparams_path):
            print(
                "Failed to locate the checkpoint's hparams.yaml file. Relying on command line args."
            )
            return
        with open(hparams_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key in config.keys():
            if key not in namespace and key != "prior_args":
                raise ValueError(f"Unknown argument in the model checkpoint: {key}")
        namespace.__dict__.update(config)
        namespace.__dict__.update(load_model=values)


def save_argparse(args, filename, exclude=None):
    import json

    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]

        ds_arg = args.get("dataset_arg")
        if ds_arg is not None and isinstance(ds_arg, str):
            args["dataset_arg"] = json.loads(args["dataset_arg"])
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def number(text):
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int
    return num_float


class MissingEnergyException(Exception):
    pass


def write_as_hdf5(files, hdf5_dataset, tile_embed=True):
    """Transform the input numpy files to hdf5 format compatible with the HDF5 Dataset class.
    The input files to this function are the same as the ones required by the Custom dataset.
    Args:
        files (dict): Dictionary of numpy input files. Must contain "pos", "z" and at least one of "y" or "neg_dy".
        hdf5_dataset (string): Path to the output HDF5 dataset.
        tile_embed (bool): Whether to tile the embeddings to match the number of samples. Default: True
    Example:
        >>> files = {}
        >>> files["pos"] = sorted(glob.glob(join(tmpdir, "coords*")))
        >>> files["z"] = sorted(glob.glob(join(tmpdir, "embed*")))
        >>> files["y"] = sorted(glob.glob(join(tmpdir, "energy*")))
        >>> files["neg_dy"] = sorted(glob.glob(join(tmpdir, "forces*")))
        >>> write_as_hdf5(files, join(tmpdir, "test.hdf5"))
    """
    import h5py

    with h5py.File(hdf5_dataset, "w") as f:
        for i in range(len(files["pos"])):
            # Create a group for each file
            coord_data = np.load(files["pos"][i], mmap_mode="r")
            embed_data = np.load(files["z"][i], mmap_mode="r").astype(int)
            group = f.create_group(str(i))
            num_samples = coord_data.shape[0]
            group.create_dataset("pos", data=coord_data)
            if tile_embed:
                group.create_dataset(
                    "types", data=np.tile(embed_data, (num_samples, 1))
                )
            else:
                group.create_dataset("types", data=embed_data)
            if "y" in files:
                energy_data = np.load(files["y"][i], mmap_mode="r")
                group.create_dataset("energy", data=energy_data)
            if "neg_dy" in files:
                force_data = np.load(files["neg_dy"][i], mmap_mode="r")
                group.create_dataset("forces", data=force_data)


def deprecated_class(cls):
    """Decorator to mark classes as deprecated."""
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def wrapped_init(self, *args, **kwargs):
        warnings.simplefilter(
            "always", DeprecationWarning
        )  # ensure all deprecation warnings are shown
        warnings.warn(
            f"{cls.__name__} is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        orig_init(self, *args, **kwargs)

    cls.__init__ = wrapped_init
    return cls


def check_logs(log_dir):
    import os
    import time

    metr_file_path = os.path.join(log_dir, "metrics.csv")
    if os.path.exists(metr_file_path):
        # we make a backup of the metrics file (rename)
        bckp_date = f'{time.strftime("%Y%m%d")}-{time.strftime("%H%M%S")}'
        os.rename(metr_file_path, metr_file_path.replace(".csv", f"_{bckp_date}.csv"))
    return
