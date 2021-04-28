# Authors: Nick Charron

from torchmdnet2.nn import TorchMD_GN
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
                               atom_ref_data.numpy())
