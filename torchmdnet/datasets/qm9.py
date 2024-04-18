# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from torch_geometric.data import Data


class QM9(QM9_geometric):
    def __init__(self, root, transform=None, label=None):
        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        assert label in label2idx, (
            "Please pass the desired property to "
            'train on via "label". Available '
            f'properties are {", ".join(label2idx)}.'
        )

        self.label = label
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        # Keep only pos, z and y in each sample
        def pre_transform(x):
            return Data(
                pos=x.pos,
                z=x.z,
                y=x.y,
            )

        super(QM9, self).__init__(
            root, transform=transform, pre_transform=pre_transform
        )

    def get_atomref(self, max_z=100):
        """Atomic energy reference values for the :py:mod:`torchmdnet.priors.Atomref` prior.

        Args:
            max_z (int): Maximum atomic number

        Returns:
            torch.Tensor: Atomic energy reference values for each element in the dataset.
        """
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        super(QM9, self).download()

    def process(self):
        super(QM9, self).process()
