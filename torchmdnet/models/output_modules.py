from abc import abstractmethod, ABCMeta
from torch_scatter import scatter
from typing import Optional
from torchmdnet.models.utils import act_class_mapping, GatedEquivariantBlock
from torchmdnet.utils import atomic_masses
from torch_scatter import scatter
import torch
from torch import nn
from warnings import warn

__all__ = ["Scalar", "DipoleMoment", "ElectronicSpatialExtent"]

def compile_check_stream_capturing():
    from torch.utils.cpp_extension import load_inline
    cpp_source = '''
#include <torch/script.h>

#if defined(WITH_CUDA)
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

bool is_stream_capturing() {
#if defined(WITH_CUDA)
  at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t cuda_stream = current_stream.stream();
  cudaStreamCaptureStatus capture_status;
  cudaError_t err = cudaStreamGetCaptureInfo(cuda_stream, &capture_status, nullptr);

  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return capture_status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive;
#else
  return false;
#endif
}

static auto registry =
  torch::RegisterOperators()
    .op("torch_extension::is_stream_capturing", &is_stream_capturing);
'''

    # Create an inline extension
    torch_extension = load_inline(
        "is_stream_capturing",
        cpp_sources=cpp_source,
        functions=["is_stream_capturing"],
        with_cuda=torch.cuda.is_available(),
        extra_cflags=["-DWITH_CUDA"] if torch.cuda.is_available() else None,
        verbose=True,
    )

compile_check_stream_capturing()
@torch.jit.script
def check_stream_capturing():
    return torch.ops.torch_extension.is_stream_capturing()


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model, reduce_op):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op
        self.dim_size = None

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def reduce(self, x, batch):
        is_capturing = (
            x.is_cuda
            and check_stream_capturing()
        )
        if not x.is_cuda or not is_capturing:
            self.dim_size = batch.max() + 1
        if is_capturing:
            assert (
                self.dim_size is not None
            ), "Warming up is needed before capturing the model into a CUDA graph"
            warn(
                "CUDA graph capture will lock the model to the current number of batches ({}). Chaning this will result in a crash".format(
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
        dtype=torch.float
    ):
        super(Scalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2, dtype=dtype),
            act_class(),
            nn.Linear(hidden_channels // 2, 1, dtype=dtype),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)


class EquivariantScalar(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
        dtype=torch.float
    ):
        super(EquivariantScalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                    dtype=dtype
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation, dtype=dtype),
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
    def __init__(self, hidden_channels, activation="silu", reduce_op="sum", dtype=torch.float):
        super(DipoleMoment, self).__init__(
            hidden_channels, activation, allow_prior_model=False, reduce_op=reduce_op, dtype=dtype
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
    def __init__(self, hidden_channels, activation="silu", reduce_op="sum", dtype=torch.float):
        super(EquivariantDipoleMoment, self).__init__(
            hidden_channels, activation, allow_prior_model=False, reduce_op=reduce_op, dtype=dtype
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
    def __init__(self, hidden_channels, activation="silu", reduce_op="sum", dtype=torch.float):
        super(ElectronicSpatialExtent, self).__init__(
            allow_prior_model=False, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2, dtype=dtype),
            act_class(),
            nn.Linear(hidden_channels // 2, 1, dtype=dtype),
        )
        atomic_mass = torch.from_numpy(atomic_masses).to(dtype)
        self.register_buffer("atomic_mass", atomic_mass)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

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
    def __init__(self, hidden_channels, activation="silu", reduce_op="sum", dtype=torch.float):
        super(EquivariantVectorOutput, self).__init__(
            hidden_channels, activation, allow_prior_model=False, reduce_op="sum", dtype=dtype
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()
