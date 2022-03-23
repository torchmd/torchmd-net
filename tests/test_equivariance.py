import torch
from torchmdnet.models.model import create_model
from utils import load_example_args


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


def test_vector_equivariance():
    torch.manual_seed(1234)
    rotate = torch.tensor(
        [
            [0.9886788, -0.1102370, 0.1017945],
            [0.1363630, 0.9431761, -0.3030248],
            [-0.0626055, 0.3134752, 0.9475304],
        ]
    )

    model = create_model(
        load_example_args(
            "equivariant-transformer", prior_model=None, output_model="VectorOutput",
        )
    )
    z = torch.ones(100, dtype=torch.long)
    pos = torch.randn(100, 3)
    batch = torch.arange(50, dtype=torch.long).repeat_interleave(2)

    y = model(z, pos, batch)[0]
    y_rot = model(z, pos @ rotate, batch)[0]
    torch.testing.assert_allclose(y @ rotate, y_rot)
