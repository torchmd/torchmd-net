from pytest import mark
import torch
from torchmdnet.models.utils import rbf_class_mapping


@mark.parametrize("name,rbf_class", list(rbf_class_mapping.items()))
def test_num_rbf(name, rbf_class, num_rbf=20):
    rbf = rbf_class(num_rbf=num_rbf)
    y = rbf(torch.linspace(0, 10, 100))
    assert y.ndim == 2, "Failed to expand the dimension."
    assert y.size(1) == num_rbf, f"Found {y.size(1)} values but expected {num_rbf}."


@mark.parametrize("cutoff_lower,cutoff_upper", [(0, 5), (1, 5), (3, 15)])
@mark.parametrize("name,rbf_class", list(rbf_class_mapping.items()))
def test_cutoff(name, rbf_class, cutoff_lower, cutoff_upper, y_tol=0.3, x_tol=0.4):
    rbf = rbf_class(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper)
    x = torch.linspace(-10, 50, 1000)
    y = rbf(x)

    assert (
        y[x < cutoff_lower - x_tol] < y_tol
    ).all(), (
        f"Found entries larger than {y_tol:.1e} below cutoff distance using {name}."
    )
    assert (
        y[x > cutoff_upper + x_tol] < y_tol
    ).all(), (
        f"Found entries larger than {y_tol:.1e} above cutoff distance using {name}."
    )
