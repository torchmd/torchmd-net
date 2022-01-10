import torch as pt
from .models.model import TorchMD_Net
from .models.torchmd_gn import TorchMD_GN


class TorchMD_GN_optimized(pt.nn.Module):

    def __init__(self, model: TorchMD_GN):

        from NNPOps.CFConv import CFConv
        from NNPOps.CFConvNeighbors import CFConvNeighbors

        super().__init__()
        self.model = model

        if self.model.cutoff_lower != 0.0:
            raise ValueError('Lower cutoff has to be 0.0')
        if self.model.neighbor_embedding:
            raise ValueError('Neighbor embedding is not supported')

        self.neighbors = CFConvNeighbors(self.model.cutoff_upper)

        self.convs = []
        for inter in self.model.interactions:
            width = 1.0 # ???
            conv = CFConv(gaussianWidth=width, activation='ssp',
                            weights1=inter.mlp[0].weight.T, biases1=inter.mlp[0].bias,
                            weights2=inter.mlp[2].weight.T, biases2=inter.mlp[2].bias)
            self.convs.append(conv)

    def forward(self, z, pos, batch):

        assert pt.all(batch == 0)

        x = self.model.embedding(z)

        # edge_index, edge_weight, _ = self.model.distance(pos, batch)
        # edge_attr = self.model.distance_expansion(edge_weight)

        # for interaction in self.model.interactions:
        #     x = x + interaction(x, edge_index, edge_weight, edge_attr)

        self.neighbors.build(pos)
        for inter, conv in zip(self.model.interactions, self.convs):
            x = inter.conv.lin1(x)
            x = conv(self.neighbors, pos, x)
            x = inter.conv.lin2(x)
            x = inter.act(x)
            x = inter.lin(x)

        return x, None, z, pos, batch

    def __repr__(self):
        return 'Optimized: ' + repr(self.model)


def optimize(model):

    assert isinstance(model, TorchMD_Net)

    if isinstance(model.representation_model, TorchMD_GN):
        model.representation_model = TorchMD_GN_optimized(model.representation_model)
    else:
        raise ValueError('Unsupported model')

    return model