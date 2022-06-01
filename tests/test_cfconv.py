import pytest
from pytest import mark
import torch as pt
from torchmdnet.models.torchmd_gn import CFConv as RefCFConv
from torchmdnet.models.utils import Distance, GaussianSmearing, ShiftedSoftplus

from NNPOps.CFConv import CFConv
from NNPOps.CFConvNeighbors import CFConvNeighbors

@mark.parametrize('device', ['cpu', 'cuda'])
@mark.parametrize(['num_atoms', 'num_filters', 'num_rbfs'], [(3, 5, 7), (3, 7, 5), (5, 3, 7), (5, 7, 3), (7, 3, 5), (7, 5, 3)])
@mark.parametrize('cutoff_upper', [5.0, 10.0])
def test_cfconv(device, num_atoms, num_filters, num_rbfs, cutoff_upper):

    if not pt.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')

    device = pt.device(device)

    # Generate random inputs
    pos = (10 * pt.rand(num_atoms, 3, dtype=pt.float32, device=device) - 5).detach()
    pos.requires_grad = True
    input = pt.rand(num_atoms, num_filters, dtype=pt.float32, device=device)

    # Construct a non-optimized CFConv object
    dist = Distance(0.0, cutoff_upper).to(device)
    rbf = GaussianSmearing(0.0, cutoff_upper, num_rbfs, trainable=False).to(device)
    net = pt.nn.Sequential(
            pt.nn.Linear(num_rbfs, num_filters),
            ShiftedSoftplus(),
            pt.nn.Linear(num_filters, num_filters))

    # Randomize linear layers
    net.requires_grad_(False)
    pt.nn.init.normal_(net[0].weight)
    pt.nn.init.normal_(net[0].bias)
    pt.nn.init.normal_(net[2].weight)
    pt.nn.init.normal_(net[2].bias)

    ref_conv = RefCFConv(num_filters, num_filters, num_filters, net, 0.0, cutoff_upper).to(device)

    # Disable the additional linear layers
    ref_conv.requires_grad_(False)
    ref_conv.lin1.weight.zero_()
    ref_conv.lin1.weight.fill_diagonal_(1)
    ref_conv.lin2.weight.zero_()
    ref_conv.lin2.weight.fill_diagonal_(1)

    # Compute with the non-optimized CFConv
    edge_index, edge_weight, _ = dist(pos, batch=None)
    edge_attr = rbf(edge_weight)
    ref_output = ref_conv(input, edge_index, edge_weight, edge_attr)
    ref_total = pt.sum(ref_output)
    ref_total.backward()
    ref_grad = pos.grad.clone()

    pos.grad.zero_()

    # Construct an optimize CFConv object
    gaussianWidth = rbf.offset[1] - rbf.offset[0]
    neighbors = CFConvNeighbors(cutoff_upper)
    conv = CFConv(gaussianWidth, 'ssp', net[0].weight.T, net[0].bias, net[2].weight.T, net[2].bias)

    # Compute with the optimized CFConv
    neighbors.build(pos)
    output = conv(neighbors, pos, input)
    total = pt.sum(output)
    total.backward()
    grad = pos.grad.clone()

    assert pt.allclose(ref_output, output, atol=5e-7)
    assert pt.allclose(ref_grad, grad, atol=5e-7)