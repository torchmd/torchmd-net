# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from torchmdnet.priors.base import BasePrior
from typing import Optional, Dict
import torch
from torch import nn, Tensor
from lightning_utilities.core.rank_zero import rank_zero_warn


class Atomref(BasePrior):
    r"""Atomref prior model.

    This prior model is used to add atomic reference values to the input features. The atomic reference values are stored in an embedding layer and are added to the input features as:

    .. math::

        x' = x + \\textrm{atomref}(z)

    where :math:`x` is the input feature tensor, :math:`z` is the atomic number tensor, and :math:`\\textrm{atomref}` is the embedding layer. The atomic reference values are stored in the embedding layer and can be trainable.

    When using this in combination with some dataset, the dataset class must implement the function `get_atomref`, which returns the atomic reference values as a tensor.

    Args:
        max_z (int, optional): Maximum atomic number to consider. If `dataset` is not `None`, this argument is ignored.
        dataset (torch_geometric.data.Dataset, optional): A dataset from which to extract the atomref values.
        trainable (bool, optional): If `False`, the atomref values are not trainable. (default: `False`)
        enable (bool, optional): If `False`, the prior is disabled. This is useful if you want to add the reference energies only during inference (or training) (default: `True`)
    """

    def __init__(self, max_z=None, dataset=None, trainable=False, enable=True):
        super().__init__()
        if max_z is None and dataset is None:
            raise ValueError("Can't instantiate Atomref prior, all arguments are None.")
        if dataset is None:
            assert max_z is not None, "max_z must be provided if dataset is None."
            atomref = torch.zeros(max_z, 1)
        else:
            atomref = dataset.get_atomref()
            if atomref is None:
                rank_zero_warn(
                    "The atomref returned by the dataset is None, defaulting to zeros with max. "
                    "atomic number 99. Maybe atomref is not defined for the current target."
                )
                atomref = torch.zeros(100, 1)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)
        self.register_buffer("initial_atomref", atomref)
        self.atomref = nn.Embedding(
            len(atomref), 1, _freeze=not trainable, _weight=atomref
        )
        self.enable = enable

    def reset_parameters(self):
        self.atomref.weight.data.copy_(self.initial_atomref)

    def get_init_args(self):
        return dict(
            max_z=self.initial_atomref.size(0),
            trainable=self.atomref.weight.requires_grad,
            enable=self.enable,
        )

    def pre_reduce(
        self,
        x: Tensor,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ):
        """Applies the stored atomref to the input as:

        .. math::

            x' = x + \\textrm{atomref}(z)

        .. note:: The atomref operation is an embedding lookup that can be trainable if the `trainable` argument is set to `False`.

        .. note:: This call becomes a no-op if the `enable` argument is set to `False`.

        Args:
            x (Tensor): Input feature tensor.
            z (Tensor): Atomic number tensor.
            pos (Tensor): Atomic positions tensor. Unused.
            batch (Tensor, optional): Batch tensor. Unused. (default: `None`).
            extra_args (Dict[str, Tensor], optional): Extra arguments. Unused. (default: `None`)


        """
        if self.enable:
            return x + self.atomref(z)
        else:
            return x


class LearnableAtomref(Atomref):
    r"""LearnableAtomref prior model.

    This prior model is used to add learned atomic reference values to the input features. The atomic reference values are learned as an embedding layer and are added to the input features as:

    .. math::

        x' = x + \\textrm{atomref}(z)

    where :math:`x` is the input feature tensor, :math:`z` is the atomic number tensor, and :math:`\\textrm{atomref}` is the embedding layer.


    Args:
        max_z (int, optional): Maximum atomic number to consider.
    """

    def __init__(self, max_z=None):
        super().__init__(max_z, trainable=True, enable=True)
