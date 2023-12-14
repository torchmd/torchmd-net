# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import yaml
from os.path import dirname, join
import torch
from torch_geometric.data import Dataset, Data


def load_example_args(model_name, remove_prior=False, config_file=None, **kwargs):
    if config_file is None:
        if model_name == "tensornet":
            config_file = join(dirname(dirname(__file__)), "examples", "TensorNet-QM9.yaml")
        else:
            config_file = join(dirname(dirname(__file__)), "examples", "ET-QM9.yaml")
    with open(config_file, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    if "precision" not in args:
        args["precision"] = 32
    args["model"] = model_name
    args["seed"] = 1234
    if remove_prior:
        args["prior_model"] = None
    for key, val in kwargs.items():
        assert key in args, f"Broken test! Unknown key '{key}'."
        args[key] = val
    return args


def create_example_batch(n_atoms=6, multiple_batches=True):
    zs = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)
    z = zs[torch.randint(0, len(zs), (n_atoms,))]

    pos = torch.randn(len(z), 3)

    batch = torch.zeros(len(z), dtype=torch.long)
    if multiple_batches:
        batch[len(batch) // 2 :] = 1
    return z, pos, batch


class DummyDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        energy=True,
        forces=True,
        atom_types=[1, 6, 7, 8, 9],
        min_atoms=3,
        max_atoms=10,
        has_atomref=False,
    ):
        super(DummyDataset, self).__init__()
        assert (
            energy or forces
        ), "The dataset must define at least one of energies and forces."

        self.z, self.pos = [], []
        self.energies = [] if energy else None
        self.forces = [] if forces else None
        for i in range(num_samples):
            num_atoms = int(torch.randint(min_atoms, max_atoms, (1,)))
            self.z.append(
                torch.tensor(atom_types)[
                    torch.randint(0, len(atom_types), (num_atoms,))
                ]
            )
            self.pos.append(torch.randn(num_atoms, 3))
            if energy:
                self.energies.append(torch.randn(1, 1))
            if forces:
                self.forces.append(torch.randn(num_atoms, 3))

        self.atomref = None
        if has_atomref:
            self.atomref = torch.randn(100, 1)

            def _get_atomref(self):
                return self.atomref

            DummyDataset.get_atomref = _get_atomref
        self.atomic_number = torch.arange(max(atom_types)+1)
        self.distance_scale = 1.0
        self.energy_scale = 1.0

    def get(self, idx):
        features = dict(z=self.z[idx].clone(), pos=self.pos[idx].clone())
        if self.energies is not None:
            features["y"] = self.energies[idx].clone()
        if self.forces is not None:
            features["neg_dy"] = self.forces[idx].clone()
        return Data(**features)

    def len(self):
        return len(self.z)
