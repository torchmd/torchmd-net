# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from abc import abstractmethod, ABCMeta
from typing import Optional
import torch
from torch import nn
from torchmdnet.models.utils import (
    act_class_mapping,
    GatedEquivariantBlock,
    scatter,
    MLP,
)
from torchmdnet.utils import atomic_masses
from torchmdnet.extensions import is_current_stream_capturing
from warnings import warn

__all__ = ["Scalar", "DipoleMoment", "ElectronicSpatialExtent"]


class OutputModel(nn.Module, metaclass=ABCMeta):
    """Base class for output models.

    Derive this class to make custom output models.
    As an example, have a look at the :py:mod:`torchmdnet.output_modules.Scalar` output model.
    """

    def __init__(self, allow_prior_model, reduce_op):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op
        self.dim_size = 0

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def reduce(self, x, batch):
        is_capturing = x.is_cuda and is_current_stream_capturing()
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
        **kwargs,
    ):
        super(Scalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
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
        **kwargs,
    ):
        super(EquivariantScalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
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
        **kwargs,
    ):
        super(DipoleMoment, self).__init__(
            hidden_channels,
            activation,
            allow_prior_model=False,
            reduce_op=reduce_op,
            dtype=dtype,
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
        **kwargs,
    ):
        super(EquivariantDipoleMoment, self).__init__(
            hidden_channels,
            activation,
            allow_prior_model=False,
            reduce_op=reduce_op,
            dtype=dtype,
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
        **kwargs,
    ):
        super(ElectronicSpatialExtent, self).__init__(
            allow_prior_model=False, reduce_op=reduce_op
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
        **kwargs,
    ):
        super(EquivariantVectorOutput, self).__init__(
            hidden_channels,
            activation,
            allow_prior_model=False,
            reduce_op="sum",
            dtype=dtype,
            **kwargs,
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()
