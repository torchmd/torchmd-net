import torch as pt
from torchmdnet.models.model import create_model
from torchmdnet.optimize import optimize

def test_gn():

    # SchNet TorchMD_GN(rbf_type='gauss', trainable_rbf=False, activation='ssp', neighbor_embedding=False)
    args = {
        'embedding_dimension': 128,
        'num_layers': 6,
        'num_rbf': 50,
        'rbf_type': 'gauss',
        'trainable_rbf': False,
        'activation': 'ssp',
        'neighbor_embedding': False,
        'cutoff_lower': 0.0,
        'cutoff_upper': 5.0,
        'max_z': 100,
        'max_num_neighbors': 32,
        'model': 'graph-network',
        'aggr': 'add',
        'derivative': True,
        'atom_filter': -1,
        'prior_model': None,
        'output_model': 'Scalar',
        'reduce_op': 'add'
    }
    model = create_model(args)
    print(model)

    num_atoms = 10
    elements = pt.randint(1, 100, (num_atoms,))
    positions = 10 * pt.rand((num_atoms, 3))
    print(elements)
    print(positions)

    energy, gradient = model(elements, positions)
    print(energy)
    print(gradient)
    ref_energy = energy.clone()
    ref_gradient = gradient.clone()

    model = optimize(model)
    print(model)

    energy, gradient = model(elements, positions)
    print(energy)
    print(gradient)
    assert pt.allclose(energy, ref_energy)
    assert pt.allclose(gradient, ref_gradient)