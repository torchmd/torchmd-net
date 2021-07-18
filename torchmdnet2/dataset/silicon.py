import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from ase.io import read, write
from copy import deepcopy


def build_self_contributions(frames, configuration_name='isolated_atom',
                            energy_name='dft_energy'):
    atomref = torch.zeros(120)
    for frame in frames:
        if configuration_name == frame.info['config_type']:
            sps = frame.get_atomic_numbers()
            atomref[sps[0]] = frame.info[energy_name]
    return atomref

class SiliconDataset(InMemoryDataset):
    r""""""

    def __init__(self, root,  transform=None, pre_transform=None,
                 pre_filter=None):

        super(SiliconDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        # Download to `self.raw_dir`.
        url_trajectory = 'https://raw.githubusercontent.com/libAtoms/silicon-testing-framework/master/models/GAP/gp_iter6_sparse9k.xml.xyz'
        path_xyz = download_url(url_trajectory, self.raw_dir)

    @property
    def raw_file_names(self):
        return ['gp_iter6_sparse9k.xml.xyz']

    @property
    def processed_file_names(self):
        return ['silicon.pt','silicon.xyz','atomrefs.pt']

    def process(self):
        frames = read(self.raw_paths[0], ':')
        atomrefs = build_self_contributions(frames, configuration_name='isolated_atom', energy_name='dft_energy')
        torch.save(atomrefs, self.processed_paths[2])

        data_list = []
        fffs = []
        for i_frame,ff in enumerate(tqdm(frames)):
            if ff.info['config_type'] == 'isolated_atom':
                continue
            try:
                energy=ff.info['dft_energy']
                forces=ff.arrays['dft_force']
                # virial = ff.info['dft_virial']
            except:
                energy=ff.info['DFT_energy']
                forces=ff.arrays['DFT_force']
                # virial = ff.info['DFT_virial']

            for sp in ff.get_atomic_numbers():
                energy -= atomrefs[sp]

            pos = torch.from_numpy(ff.get_positions().reshape(-1, 3))
            z = torch.from_numpy(ff.get_atomic_numbers())
            forces = torch.from_numpy(forces.reshape(-1, 3))
            pbc = torch.from_numpy(ff.get_pbc().reshape(1, 3))
            cell = torch.from_numpy(ff.get_cell().array.reshape(1, 3, 3))

            data = Data(z=z, pos=pos, energy=energy, forces=forces, idx=i_frame,
                        cell=cell,pbc=pbc,
                        name=ff.info['config_type'], n_atoms=len(ff))

            if self.pre_filter != None and not self.pre_filter(data):
                continue
            if self.pre_transform != None:
                data = self.pre_transform(data)
            data_list.append(data)

            fff = deepcopy(ff)
            fff.info['dft_energy'] = energy.numpy()
            fff.arrays['dft_force'] = forces.numpy()

            fffs.append(fff)

        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])
        write(self.processed_paths[1], fffs)
