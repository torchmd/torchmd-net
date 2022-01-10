import torch as pt
from .models.model import TorchMD_Net
from .models.torchmd_gn import TorchMD_GN


class TorchMD_GN_optimized(pt.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, z, pos, batch):

        x = self.model.embedding(z)

        edge_index, edge_weight, _ = self.model.distance(pos, batch)
        edge_attr = self.model.distance_expansion(edge_weight)

        if self.model.neighbor_embedding is not None:
            x = self.model.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        for interaction in self.model.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)

        return x, None, z, pos, batch

    def __repr__(self):
        return repr(self.model)


def optimize(model):

    assert isinstance(model, TorchMD_Net)

    if isinstance(model.representation_model, TorchMD_GN):
        model.representation_model= TorchMD_GN_optimized(model.representation_model)
    else:
        raise ValueError('Unsupported model')

    return model