import torch
from torchmdnet.models.model import create_model
from utils import load_example_args, create_example_batch
from torchmdnet.models.output_modules import OutputModel
import pytorch_lightning as pl
import pytest

def test_scalar_invariance():
    torch.manual_seed(1234)
    rotate = torch.tensor(
        [
            [0.9886788, -0.1102370, 0.1017945],
            [0.1363630, 0.9431761, -0.3030248],
            [-0.0626055, 0.3134752, 0.9475304],
        ]
    )

    model = create_model(load_example_args("equivariant-transformer", prior_model=None))
    z = torch.ones(100, dtype=torch.long)
    pos = torch.randn(100, 3)
    batch = torch.arange(50, dtype=torch.long).repeat_interleave(2)

    y = model(z, pos, batch)[0]
    y_rot = model(z, pos @ rotate, batch)[0]
    torch.testing.assert_allclose(y, y_rot)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vector_equivariance(dtype):
    torch.manual_seed(1234)
    rotate = torch.tensor(
        [
            [0.9886788, -0.1102370, 0.1017945],
            [0.1363630, 0.9431761, -0.3030248],
            [-0.0626055, 0.3134752, 0.9475304],
        ]
    ).to(dtype)

    model = create_model(
        load_example_args(
            "equivariant-transformer",
            prior_model=None,
            output_model="VectorOutput",
            dtype=dtype,
        )
    )
    z = torch.ones(100, dtype=torch.long)
    pos = torch.randn(100, 3).to(dtype)
    batch = torch.arange(50, dtype=torch.long).repeat_interleave(2)

    y = model(z, pos, batch)[0]
    y_rot = model(z, pos @ rotate, batch)[0]
    torch.testing.assert_allclose(y @ rotate, y_rot)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensornet_energy_invariance(dtype):
    torch.manual_seed(1234)
    pl.seed_everything(1234)

    # create model and sample batch
    args = load_example_args(
        "tensornet",
        remove_prior=True,
        output_model="Scalar",
        derivative=True,
        dtype=dtype,
    )
    model = create_model(args)
    natoms = 10
    z = torch.ones(natoms, dtype=torch.long)
    pos = torch.randn(natoms, 3).to(dtype)
    batch = torch.zeros_like(z)
    pos.to(dtype)
    batch = torch.zeros_like(batch)
    # run step
    y, _ = model(z, pos, batch)
    alpha = torch.rand(1).to(dtype) * 2 * 3.141592653589793
    beta = torch.rand(1).to(dtype) * 2 * 3.141592653589793
    gamma = torch.rand(1).to(dtype) * 2 * 3.141592653589793
    Rx = torch.tensor( [ [1, 0, 0], [0, torch.cos(alpha), -torch.sin(alpha)], [0, torch.sin(alpha), torch.cos(alpha)] ])
    Ry = torch.tensor( [ [torch.cos(beta), 0, torch.sin(beta)], [0, 1, 0], [-torch.sin(beta), 0, torch.cos(beta)] ] )
    Rz = torch.tensor( [ [torch.cos(gamma), -torch.sin(gamma), 0], [torch.sin(gamma), torch.cos(gamma), 0], [0, 0, 1] ] )
    rotate = (Rx @ Ry @ Rz).to(dtype)
    y_rot, _ = model(z, pos @ rotate, batch)
    torch.testing.assert_allclose(y, y_rot, rtol=1e-13 if dtype == torch.float64 else 1e-6, atol= 0)


from torch_scatter import scatter
class TensorOutput(OutputModel):
    """ Output model for tensor properties.
    Only compatible with TensorNet

    """
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
        dtype=torch.float
    ):
        super(TensorOutput, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        self.reset_parameters()

    def reduce(self, input, batch):
        I, A, S =  input
        I = scatter(I.sum(-3), batch, dim=0, reduce=self.reduce_op)
        A = scatter(A.sum(-3), batch, dim=0, reduce=self.reduce_op)
        S = scatter(S.sum(-3), batch, dim=0, reduce=self.reduce_op)
        return I+A+S

    def reset_parameters(self):
        pass

    def pre_reduce(self, x, v, z, pos, batch):
        return v

    def post_reduce(self, x):
        return x

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensornet_equivariance(dtype):
    torch.manual_seed(1234)
    pl.seed_everything(1234)

    # create model and sample batch
    args = load_example_args(
        "tensornet",
        remove_prior=True,
        output_model="Scalar",
        derivative=False,
        dtype=dtype,
    )
    model = create_model(args)
    model.output_model = TensorOutput(args["embedding_dimension"],
        activation=args["activation"],
        reduce_op=args["reduce_op"],
        dtype=args["dtype"])

    natoms = 10
    z = torch.ones(natoms, dtype=torch.long)
    pos = torch.randn(natoms, 3).to(dtype)
    batch = torch.zeros_like(z)
    pos.to(dtype)
    batch = torch.zeros_like(batch)
    # run step
    X = model(z, pos, batch)[0]

    alpha = torch.rand(1).to(dtype) * 2 * 3.141592653589793
    beta = torch.rand(1).to(dtype) * 2 * 3.141592653589793
    gamma = torch.rand(1).to(dtype) * 2 * 3.141592653589793
    Rx = torch.tensor( [ [1, 0, 0], [0, torch.cos(alpha), -torch.sin(alpha)], [0, torch.sin(alpha), torch.cos(alpha)] ])
    Ry = torch.tensor( [ [torch.cos(beta), 0, torch.sin(beta)], [0, 1, 0], [-torch.sin(beta), 0, torch.cos(beta)] ] )
    Rz = torch.tensor( [ [torch.cos(gamma), -torch.sin(gamma), 0], [torch.sin(gamma), torch.cos(gamma), 0], [0, 0, 1] ] )
    rotate = (Rx @ Ry @ Rz).to(dtype)

    Xrot = model(z, pos @ rotate, batch)[0]
    torch.testing.assert_allclose(rotate.t()@(X @ rotate), Xrot, rtol=5e-13 if dtype == torch.float64 else 5e-5, atol= 0)
