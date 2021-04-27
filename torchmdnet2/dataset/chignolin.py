import os
import os.path as osp
from glob import glob
from os.path import isfile, join
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)

import mdtraj

from ..nn import BaselineModel, RepulsionLayer, HarmonicLayer
from ..geometry import GeometryFeature, GeometryStatistics
from ..utils import tqdm


class ChignolinDataset(InMemoryDataset):
    r""""""

    def __init__(self, root,  transform=None, pre_transform=None,
                 pre_filter=None):
        self.temperature = 350 # K
        super(ChignolinDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.traj = mdtraj.load(self.processed_paths[1])

    def download(self):
        # Download to `self.raw_dir`.
        url_trajectory = 'http://pub.htmd.org/chignolin_trajectories.tar.gz'
        url_forces = 'http://pub.htmd.org/chignolin_forces_nowater.tar.gz'
        url_coords = 'http://pub.htmd.org/chignolin_coords_nowater.tar.gz'
        url_inputs = 'http://pub.htmd.org/chignolin_generators.tar.gz'
        # path_trajectory = download_url(url_trajectory, self.raw_dir)
        path_inputs = download_url(url_inputs, self.raw_dir)
        path_coord = download_url(url_coords, self.raw_dir)
        path_forces = download_url(url_forces, self.raw_dir)

    @property
    def raw_file_names(self):
        return ['chignolin_generators.tar.gz', 'chignolin_forces_nowater.tar.gz', 'chignolin_coords_nowater.tar.gz']

    @property
    def processed_file_names(self):
        return ['chignolin.pt', 'chignolin.pdb']

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
    def get_data_filenames(coord_dir, force_dir):
        tags = [os.path.basename(fn).replace('chig_coor_','').replace('.npy','') for fn in  os.listdir(coord_dir)]

        coord_fns = {}
        for tag in tags:
            fn = f'chig_coor_{tag}.npy'
            coord_fns[tag] = join(coord_dir, fn)
        forces_fns = {}
        for tag in tags:
            fn = f'chig_force_{tag}.npy'
            forces_fns[tag] = join(force_dir, fn)
        return coord_fns,forces_fns

    def get_baseline_model(self, data=None, n_beads=None):
        if data is None:
            data = self.data
        if n_beads is None:
            n_beads = data.n_beads[0]
        priors = []
        coordinates = data.pos.cpu().detach().numpy().reshape((-1, n_beads, 3))
        stats = GeometryStatistics(coordinates, temperature=self.temperature, backbone_inds='all', get_all_distances=True,
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

        baseline_model = BaselineModel(geometry_feature, priors, n_beads,
                                        beta=stats.beta)

        return baseline_model

    @staticmethod
    def _remove_baseline_forces(data, slices, baseline_model):
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
        slices['baseline_forces'] = slices['forces']
        return data

    def process(self):
        # extract files
        for fn in self.raw_paths:
            extract_tar(fn, self.raw_dir, mode='r:gz')
        coord_dir = join(self.raw_dir, 'coords_nowater')
        force_dir = join(self.raw_dir, 'forces_nowater')

        topology_fn = join(self.raw_dir,'chignolin_50ns_0/structure.pdb')
        traj = mdtraj.load(topology_fn).remove_solvent()
        traj.save(self.processed_paths[1])

        topology = traj.topology
        embeddings, cg_matrix, cg_mapping = self.get_cg_mapping(topology)
        n_beads = cg_matrix.shape[0]
        embeddings = np.array(embeddings, dtype=np.int64)

        coord_fns, forces_fns = self.get_data_filenames(coord_dir, force_dir)

        f_proj = np.dot(np.linalg.inv(np.dot(cg_matrix,cg_matrix.T)),cg_matrix)

        data_list = []
        ii_frame = 0
        for i_traj, tag in enumerate(tqdm(coord_fns, desc='Load Dataset')):
            forces = np.load(forces_fns[tag])
            cg_forces = np.array(np.einsum('mn, ind-> imd', f_proj, forces), dtype=np.float32)

            coords = np.load(coord_fns[tag])
            cg_coords = np.array(np.einsum('mn, ind-> imd',cg_matrix, coords), dtype=np.float32)

            n_frames = cg_coords.shape[0]

            for i_frame in range(n_frames):
                pos = torch.from_numpy(cg_coords[i_frame].reshape(n_beads, 3))
                z = torch.from_numpy(embeddings)
                force = torch.from_numpy(cg_forces[i_frame].reshape(n_beads, 3))

                data = Data(z=z, pos=pos, forces=force, idx=ii_frame,
                            name='chignolin', n_beads=n_beads, traj_idx=i_traj)
                if self.pre_filter != None and not self.pre_filter(data):
                    continue
                if self.pre_transform != None:
                    data = self.pre_transform(data)
                data_list.append(data)
                ii_frame += 1

        datas, slices = self.collate(data_list)

        baseline_model = self.get_baseline_model(datas, n_beads)
        self._remove_baseline_forces(datas, slices, baseline_model)

        torch.save((datas, slices), self.processed_paths[0])