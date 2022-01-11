import torch as pt
from .models.model import TorchMD_Net
from .models.torchmd_gn import TorchMD_GN


class TorchMD_GN_optimized(pt.nn.Module):

    def __init__(self, model):

        from NNPOps.CFConv import CFConv
        from NNPOps.CFConvNeighbors import CFConvNeighbors

        super().__init__()
        self.model = model

        if self.model.cutoff_lower != 0.0:
            raise ValueError('Lower cutoff has to be 0.0')
        if self.model.neighbor_embedding:
            raise ValueError('Neighbor embedding is not supported')

        self.neighbors = CFConvNeighbors(self.model.cutoff_upper)

        offset = self.model.distance_expansion.offset
        width = offset[1] - offset[0]
        self.convs = [CFConv(gaussianWidth=width, activation='ssp',
                             weights1=inter.mlp[0].weight.T, biases1=inter.mlp[0].bias,
                             weights2=inter.mlp[2].weight.T, biases2=inter.mlp[2].bias)
                      for inter in self.model.interactions]

    def forward(self, z, pos, batch):

        assert pt.all(batch == 0)

        x = self.model.embedding(z)

        self.neighbors.build(pos)
        for inter, conv in zip(self.model.interactions, self.convs):
            y = inter.conv.lin1(x)
            y = conv(self.neighbors, pos, y)
            y = inter.conv.lin2(y)
            y = inter.act(y)
            x = x + inter.lin(y)

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