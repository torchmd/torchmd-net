# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from abc import abstractmethod, ABCMeta
from typing import Optional
import torch
from torch import nn
from torchmdnet.models.utils import GatedEquivariantBlock, scatter, MLP
from torchmdnet.utils import atomic_masses
from warnings import warn

__all__ = ["Scalar", "DipoleMoment", "ElectronicSpatialExtent"]


class OutputModel(nn.Module, metaclass=ABCMeta):
    """Base class for output models.

    Derive this class to make custom output models.
    As an example, have a look at the :py:mod:`torchmdnet.output_modules.Scalar` output model.
    """

    def __init__(self, allow_prior_model, reduce_op, static_shapes=False):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op
        self.static_shapes = static_shapes
        self.dim_size = 0

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def reduce(self, x, batch):
        # torch.compile and torch.export don't support .item() calls during tracing
        # The model should be warmed up before compilation to set the correct dim_size
        if torch.compiler.is_compiling():
            pass
        elif torch.jit.is_scripting():
            # TorchScript doesn't support torch.cuda.is_current_stream_capturing()
            # For CPU, always update dim_size (no CUDA graphs on CPU)
            # For CUDA with static_shapes, only update once (first call sets dim_size for CUDA graph capture)
            # For CUDA without static_shapes, always update (dynamic batch sizes)
            if not x.is_cuda or not self.static_shapes or self.dim_size == 0:
                self.dim_size = int(batch.max().item() + 1)
        else:
            is_capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()
            if not x.is_cuda or not is_capturing:
                self.dim_size = int(batch.max().item() + 1)
            if is_capturing:
                assert (
                    self.dim_size > 0
                ), "Warming up is needed before capturing the model into a CUDA graph"
                warn(
                    "CUDA graph capture will lock the batch to the current number of samples ({}). Changing this will result in a crash".format(
                        self.dim_size
                    )
                )
        return scatter(x, batch, dim=0, dim_size=self.dim_size, reduce=self.reduce_op)

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
        dtype=torch.float,
        static_shapes=False,
        **kwargs,
    ):
        super(Scalar, self).__init__(
            allow_prior_model=allow_prior_model,
            reduce_op=reduce_op,
            static_shapes=static_shapes,
        )
        self.output_network = MLP(
            in_channels=hidden_channels,
            out_channels=1,
            hidden_channels=hidden_channels // 2,
            activation=activation,
            num_hidden_layers=kwargs.get("num_layers", 0),
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.output_network.reset_parameters()

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)


class EquivariantScalar(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
        dtype=torch.float,
        static_shapes=False,
        **kwargs,
    ):
        super(EquivariantScalar, self).__init__(
            allow_prior_model=allow_prior_model,
            reduce_op=reduce_op,
            static_shapes=static_shapes,
        )
        if kwargs.get("num_layers", 0) > 0:
            warn("num_layers is not used in EquivariantScalar")
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                    dtype=dtype,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2, 1, activation=activation, dtype=dtype
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0


class DipoleMoment(Scalar):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        reduce_op="sum",
        dtype=torch.float,
        static_shapes=False,
        **kwargs,
    ):
        super(DipoleMoment, self).__init__(
            hidden_channels,
            activation,
            allow_prior_model=False,
            reduce_op=reduce_op,
            dtype=dtype,
            static_shapes=static_shapes,
            **kwargs,
        )
        atomic_mass = torch.from_numpy(atomic_masses).to(dtype)
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class EquivariantDipoleMoment(EquivariantScalar):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        reduce_op="sum",
        dtype=torch.float,
        static_shapes=False,
        **kwargs,
    ):
        super(EquivariantDipoleMoment, self).__init__(
            hidden_channels,
            activation,
            allow_prior_model=False,
            reduce_op=reduce_op,
            dtype=dtype,
            static_shapes=static_shapes,
            **kwargs,
        )
        atomic_mass = torch.from_numpy(atomic_masses).to(dtype)
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x + v.squeeze()

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class ElectronicSpatialExtent(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        reduce_op="sum",
        dtype=torch.float,
        static_shapes=False,
        **kwargs,
    ):
        super(ElectronicSpatialExtent, self).__init__(
            allow_prior_model=False, reduce_op=reduce_op, static_shapes=static_shapes
        )
        self.output_network = MLP(
            in_channels=hidden_channels,
            out_channels=1,
            hidden_channels=hidden_channels // 2,
            activation=activation,
            num_hidden_layers=kwargs.get("num_layers", 0),
            dtype=dtype,
        )
        atomic_mass = torch.from_numpy(atomic_masses).to(dtype)
        self.register_buffer("atomic_mass", atomic_mass)

        self.reset_parameters()

    def reset_parameters(self):
        self.output_network.reset_parameters()

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)

        x = torch.norm(pos - c[batch], dim=1, keepdim=True) ** 2 * x
        return x


class EquivariantElectronicSpatialExtent(ElectronicSpatialExtent):
    pass


class EquivariantVectorOutput(EquivariantScalar):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        reduce_op="sum",
        dtype=torch.float,
        static_shapes=False,
        **kwargs,
    ):
        super(EquivariantVectorOutput, self).__init__(
            hidden_channels,
            activation,
            allow_prior_model=False,
            reduce_op="sum",
            dtype=dtype,
            static_shapes=static_shapes,
            **kwargs,
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()


def _exp_cutoff(d, rc: float):
    """short range cutoff to damp close range coulomb interaction

    Implementation copied form aimnetcentral
    """
    fc = (
        torch.exp(-1.0 / (1.0 - (d / rc).clamp(0, 1.0 - 1e-6).pow(2)))
        / 0.36787944117144233
    )
    return fc


def _triu_indices(x, batch):
    N = x.shape[0]
    i, j = torch.triu_indices(N, N, 1, device=x.device, dtype=torch.long).unbind(0)
    mask = batch[i] == batch[j]
    i = i[mask]
    j = j[mask]
    return torch.stack([i, j])


class ScalarPlusWeightedCoulomb(OutputModel):
    """normal scalar output plus a coulomb interaction using predicted charges

    It assumes the input features are shape [N, hidden_channels + q_channels]
    where q_channels are the predicted partial charges (can be 1 or length N)
    If more than 1 the coulomb interaction is computed seperately for each charge
    channel and then a weighted mean is taken of the resulting energies.

    Implementation modifed from aimnetcentral
    """

    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
        dtype=torch.float,
        static_shapes=False,
        **kwargs,
    ):
        super(ScalarPlusWeightedCoulomb, self).__init__(
            allow_prior_model=allow_prior_model,
            reduce_op=reduce_op,
            static_shapes=static_shapes,
        )
        self.hidden_channels = hidden_channels
        self.output_network = MLP(
            in_channels=hidden_channels,
            out_channels=1,
            hidden_channels=hidden_channels // 2,
            activation=activation,
            num_hidden_layers=kwargs.get("num_hidden_layers", 0),
            dtype=dtype,
        )

        self.q_dim = kwargs["q_dim"]
        self.num_interaction_layers = kwargs["num_layers"]
        self.layer_weights = kwargs["q_weights"]

        assert len(self.layer_weights) == self.num_interaction_layers + 1

        _w = torch.zeros((self.num_interaction_layers + 1, self.q_dim), dtype=dtype)
        for i in range(self.num_interaction_layers + 1):
            _w[i, :] = torch.tensor(self.layer_weights[i], dtype=dtype)

        _w = _w.flatten()

        self.register_buffer("qweights", _w)

        # from ase.units from aimnet2
        _Hartree = 27.211386024367243
        _half_Hartree = 0.5 * _Hartree
        _Bohr = 0.5291772105638411
        self._factor = _half_Hartree * _Bohr

        self.edge_index = torch.empty(2, 0)

        self.reset_parameters()

    def reset_parameters(self):
        self.output_network.reset_parameters()

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        # we assume the charges have been added at the end of x!
        charges = x[:, self.hidden_channels :]

        if not self.static_shapes:  # check we have the expected number of charges
            assert charges.shape[1] == self.qweights.shape[0]

        x = x[:, : self.hidden_channels]
        x = self.output_network(x)  # energy per atom from main MLPs

        # torch.compile and torch.export don't support .item() calls during tracing
        # The model should be warmed up before compilation to set the correct dim_size
        if torch.compiler.is_compiling():
            assert self.edge_index.shape[1] > 0

        elif torch.jit.is_scripting():
            # TorchScript doesn't support torch.cuda.is_current_stream_capturing()
            # For CPU, always update dim_size (no CUDA graphs on CPU)
            # For CUDA with static_shapes, only update once (first call sets dim_size for CUDA graph capture)
            # For CUDA without static_shapes, always update (dynamic batch sizes)
            if not x.is_cuda or not self.static_shapes or self.edge_index.shape[1] == 0:
                self.edge_index = _triu_indices(x, batch)
        else:
            is_capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()
            if not x.is_cuda or not is_capturing:
                self.edge_index = _triu_indices(x, batch)

            if is_capturing:
                assert (
                    self.edge_index is not None
                ), "Warming up is needed before capturing the model into a CUDA graph"
                warn(
                    "CUDA graph capture will lock the batch to the current number of samples ({}). Changing this will result in a crash".format(
                        self.dim_size
                    )
                )

        q_i = charges[self.edge_index[0]]
        q_j = charges[self.edge_index[1]]

        d_ij = torch.linalg.norm(
            pos[self.edge_index[0]] - pos[self.edge_index[1]], dim=-1
        )

        # charge product
        q_ij = q_i * q_j

        # short range damping, hyperparameter is from aimnet2
        fc = 1.0 - _exp_cutoff(d_ij, 4.6)

        # per edge Coulomb energy with short range damping
        e_ij = fc.unsqueeze(-1) * q_ij / d_ij.unsqueeze(-1)

        # scale with Coulomb constant in eV and Angstrom unit system!
        e_ij = self._factor * e_ij

        # now do the weighted mean over features
        e_ij = torch.sum(e_ij * self.qweights.unsqueeze(0), dim=-1) / torch.sum(
            self.qweights
        )

        # sum into nodes
        e_i = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        e_i = e_i.index_add(0, self.edge_index[0], e_ij)
        e_i = e_i.index_add(0, self.edge_index[1], e_ij)

        # add to normal per atom output with correct shape
        x = x + e_i.unsqueeze(-1)

        return x
