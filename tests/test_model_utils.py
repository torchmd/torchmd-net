import torch
from torch.autograd import grad
from torchmdnet.models.model import create_model
from utils import load_example_args


def test_gated_eq_gradients():
    model = create_model(
        load_example_args(
            "equivariant-transformer", prior_model=None, cutoff_upper=5, derivative=True
        )
    )

    # generate example where one atom is outside the cutoff radius of all others
    z = torch.tensor([1, 1, 8])
    pos = torch.tensor([[0, 0, 0], [0, 1, 0], [10, 0, 0]], dtype=torch.float)

    _, forces = model(z, pos)

    # compute gradients of forces with respect to the model's emebdding weights
    deriv = grad(forces.sum(), model.representation_model.embedding.weight)[0]
    assert (
        not deriv.isnan().any()
    ), "Encountered NaN gradients while backpropagating the force loss"
