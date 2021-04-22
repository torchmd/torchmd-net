import os
import os.path as osp
from glob import glob
from os.path import isfile, join
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)

import mdtraj

from ..nn import BaselineModel, RepulsionLayer, HarmonicLayer
from ..geometry import GeometryFeature, GeometryStatistics



AA2INT = {'ALA':1,
         'GLY':2,
         'PHE':3,
         'TYR':4,
         'ASP':5,
         'GLU':6,
         'TRP':7,
         'PRO':8,
         'ASN':9,
         'GLN':10,
         'HIS':11,
         'HSD':11,
         'HSE':11,
         'SER':12,
         'THR':13,
         'VAL':14,
         'MET':15,
         'CYS':16,
         'NLE':17,
         'ARG':18,
         'LYS':19,
         'LEU':20,
         'ILE':21
         }


class ChignolinDataset(InMemoryDataset):
    r""""""

    def __init__(self, root,  transform=None, pre_transform=None,
                 pre_filter=None):

        super(ChignolinDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        # Download to `self.raw_dir`.
        url_trajectory = 'http://pub.htmd.org/chignolin_trajectories.tar.gz'
        url_forces = 'http://pub.htmd.org/chignolin_forces_nowater.tar.gz'

        path_trajectory = download_url(url_trajectory, self.raw_dir)
        path_forces = download_url(url_forces, self.raw_dir)

    @property
    def raw_file_names(self):
        return ['chignolin_trajectories.tar.gz', 'chignolin_forces_nowater.tar.gz']

    @property
    def processed_file_names(self):
        return 'chignolin.pt'

    @staticmethod
    def get_cg_mapping(topology):
        cg_mapping = OrderedDict()
        for i in range(topology.n_atoms):
            atom = topology.atom(i)
            if 'CA' == atom.name:
                cg_mapping[i] = atom.residue.name
        # terminal beads are treated differently from others
        aa = list(cg_mapping)
        cg_mapping[aa[0]] += '-terminal'
        cg_mapping[aa[-1]] += '-terminal'

        n_beads = len(cg_mapping)
        n_atoms = topology.n_atoms
        cg_matrix = np.zeros((n_beads,n_atoms))
        residue_mapping = {v:k for k,v in enumerate(np.unique([v for v in cg_mapping.values()]))}
        embeddings = np.array([residue_mapping[v] for v in cg_mapping.values()])
        embeddings
        for i,(k,v) in enumerate(cg_mapping.items()):
            cg_matrix[i, k] = 1
        return embeddings,cg_matrix,cg_mapping

    @staticmethod
    def get_data_filenames(traj_dir, force_dir):
        tags = [os.path.basename(fn) for fn in  os.listdir(traj_dir) if not isfile(join(traj_dir, fn))]
        traj_fns = {}
        for tag in tags:
            aa = glob(join(traj_dir,f'{tag}/*.xtc'))
            assert len(aa) == 1, aa
            traj_fns[tag] = aa[0]
        forces_fns = {}
        for tag in tags:
            fn = f'chig_force_{tag}.npy'
            forces_fns[tag] = join(force_dir, fn)
        return traj_fns,forces_fns

    def get_baseline_model(self, data=None, n_beads=None):
        if data is None:
            data = self.data
        if n_beads is None:
            n_beads = data.n_beads[0]
        priors = []
        coordinates = data.pos.cpu().detach().numpy().reshape((-1, n_beads, 3))
        stats = GeometryStatistics(coordinates, backbone_inds='all', get_all_distances=True,
                          get_backbone_angles=True, get_backbone_dihedrals=True)

        # estimate the harmonic bond parameters
        bond_list, bond_keys = stats.get_prior_statistics(features='Bonds', as_list=True)
        bond_indices = stats.return_indices('Bonds')
        priors += [HarmonicLayer(bond_indices, bond_list)]
        # estimate the harmonic angle parameters
        angle_list, angle_keys = stats.get_prior_statistics(features='Angles', as_list=True)
        angle_indices = stats.return_indices('Angles')
        priors += [HarmonicLayer(angle_indices, angle_list)]

        # repulsion between all beads
        repul_distances = [i for i in stats.descriptions['Distances']]
        repul_idx = stats.return_indices(repul_distances)  # Indices of beads
        repul_list = [{'ex_vol': 4.5, "exp": 6.} for i in range(len(repul_distances))]
        priors += [RepulsionLayer(repul_idx, repul_list)]

        geometry_feature = GeometryFeature(feature_tuples='all_backbone', n_beads=n_beads)

        baseline_model = BaselineModel(geometry_feature, priors, n_beads)

        return baseline_model

    @staticmethod
    def _remove_baseline_forces(data, baseline_model):
        data.pos.requires_grad_()
        n_frames = data.idx.shape[0]
        baseline_energy = baseline_model(data.pos, n_frames)
        baseline_force = -torch.autograd.grad(baseline_energy,
                                    data.pos,
                                     grad_outputs=torch.ones_like(baseline_energy),
                                    create_graph=False,
                                    retain_graph=False)[0]
        data.pos.requires_grad = False
        data.forces -= baseline_force
        data.baseline_forces = baseline_force

        return data

    def process(self):
        # extract files
        for fn in self.raw_paths:
            extract_tar(fn, self.raw_dir, mode='r:gz')
        traj_dir = join(self.raw_dir, 'filtered')
        force_dir = join(self.raw_dir, 'forces_nowater')

        topology_fn = join(traj_dir,'filtered.pdb')
        topology = mdtraj.load(topology_fn).topology

        embeddings, cg_matrix, cg_mapping = self.get_cg_mapping(topology)
        n_beads = cg_matrix.shape[0]
        embeddings = np.array(embeddings, dtype=np.int64)

        traj_fns, forces_fns = self.get_data_filenames(traj_dir, force_dir)

        f_proj = np.dot(np.linalg.inv(np.dot(cg_matrix,cg_matrix.T)),cg_matrix)

        data_list = []
        ii_frame = 0
        for tag in tqdm(traj_fns, desc='Load Dataset'):
            forces = np.load(forces_fns[tag])
            # the chignolin trajectory as 2 ions at the end while the forces don't record these atoms
            cg_forces = np.array(np.einsum('mn, ind-> imd', f_proj[:,:-2], forces), dtype=np.float32)
            cg_coords = []

            for chunk in mdtraj.iterload(traj_fns[tag], chunk=100, top=topology_fn):
                cg_coord = np.array(np.einsum('mn, ind-> imd',
                                                 cg_matrix, chunk.xyz), dtype=np.float32)
                cg_coords.append(cg_coord)
            cg_coords = np.vstack(cg_coords)
            assert cg_coords.shape == cg_forces.shape

            for i_frame in range(chunk.n_frames):
                pos = torch.from_numpy(cg_coords[i_frame].reshape(n_beads, 3))
                z = torch.from_numpy(embeddings)
                force = torch.from_numpy(cg_forces[i_frame].reshape(n_beads, 3))

                data = Data(z=z, pos=pos, forces=force, idx=ii_frame,
                            name='chignolin', n_beads=n_beads)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                ii_frame += 1

        datas, slices = self.collate(data_list)

        baseline_model = self.get_baseline_model(datas, n_beads)
        datas = self._remove_baseline_forces(datas, baseline_model)

        torch.save((datas, slices), self.processed_paths[0])