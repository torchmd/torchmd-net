# Authors: Nick Charron

from torchmdnet2.nn import TorchMD_GN, GraphNormMSE
import numpy as np
import torch

def test_cfconv_agg_mode():
    # Tests to make sure that top level kwargs for CFConv
    # are correctly passed down to each CFConv in the model

    # Test each aggregation option offered via torch_scatter
    for aggr in ['add', 'mean', 'max', None]:
        model = TorchMD_GN(cfconv_aggr=aggr)
        for block in model.interactions:
            conv = block.conv
            assert conv.aggr == aggr


def test_embedding_size():
    # Tests to make sure the embedding layer is instanced to the
    # the specified size

    dictionary_size = np.random.randint(low=3, high=5)
    model = TorchMD_GN(embedding_size=dictionary_size)
    assert model.embedding.num_embeddings == dictionary_size

    # Next, we test if an atomref is initialized

    atom_ref_data = torch.randn(size=(dictionary_size, 1))
    model = TorchMD_GN(embedding_size=dictionary_size,
                       atomref=atom_ref_data)
    assert model.embedding.num_embeddings == dictionary_size
    assert model.atomref.weight.shape == atom_ref_data.shape
    np.testing.assert_allclose(model.atomref.weight.detach().numpy(),
                               atom_ref_data.numpy(),
                               rtol=1e-12)


def test_graph_norm_loss():
    # Tests to make sure that the node noramlized loss works
    # as expected

    n_batches = np.random.randint(low=5, high=25)
    mol_sizes = np.random.randint(low=10, high=50, size=(n_batches,))
    graph_labels = []
    size_labels = []
    for i, size in enumerate(mol_sizes):
        for _ in range(size):
            graph_labels.append(i)
            size_labels.append(size)

    n_examples = int(np.sum(mol_sizes))
    random_force_data = np.random.randn(n_examples, 3)
    random_force_labels = np.random.randn(n_examples, 3)

    loss_fn = GraphNormMSE()
    torch_loss = loss_fn(torch.tensor(random_force_data),
                         torch.tensor(random_force_labels),
                         torch.tensor(graph_labels))

    numpy_loss = random_force_data - random_force_labels
    numpy_loss = numpy_loss * np.array(size_labels)[:, None]
    numpy_loss = (numpy_loss ** 2).mean()

    np.testing.assert_allclose(numpy_loss, torch_loss.numpy(),
                               rtol=1e-12)





