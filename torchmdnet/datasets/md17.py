import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np


class MD17(InMemoryDataset):

    raw_url = 'http://www.quantum-machine.org/gdml/data/npz/'

    molecule_files = dict(
        aspirin='aspirin_dft.npz',
        azobenzene='azobenzene_dft.npz',
        benzene='benzene_dft.npz',
        ethanol='ethanol_dft.npz',
        malonaldehyde='malonaldehyde_dft.npz',
        naphthalene='naphthalene_dft.npz',
        paracetamol='paracetamol_dft.npz',
        salicylic_acid='salicylic_dft.npz',
        toluene='toluene_dft.npz',
        uracil='uracil_dft.npz',
    )

    molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, ('Please pass the desired molecule to '
                                         'train on via "dataset_arg". Available '
                                         f'molecules are {", ".join(MD17.molecules)}.')

        self.molecule = dataset_arg
        super(MD17, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [MD17.molecule_files[self.molecule]]

    @property
    def processed_file_names(self):
        return [f'md17-{self.molecule}.pt']

    def download(self):
        download_url(MD17.raw_url + self.raw_file_names[0], self.raw_dir)

    def process(self):
        data_npz = np.load(self.raw_paths[0])
        z = torch.from_numpy(data_npz['z']).long()
        positions = torch.from_numpy(data_npz['R']).float()
        energies = torch.from_numpy(data_npz['E']).float()
        forces = torch.from_numpy(data_npz['F']).float()
        
        samples = []
        for pos, y, dy in zip(positions, energies, forces):
            samples.append(Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
