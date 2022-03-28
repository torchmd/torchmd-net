from typing import Optional, Tuple
import torch as pt
from NNPOps.CFConv import CFConv
from NNPOps.CFConvNeighbors import CFConvNeighbors

from .models.model import TorchMD_Net
from .models.torchmd_gn import TorchMD_GN


class TorchMD_GN_optimized(pt.nn.Module):
    def __init__(self, model):

        if model.rbf_type != "gauss":
            raise ValueError('Only rbf_type="gauss" is supproted')
        if model.trainable_rbf:
            raise ValueError("trainalbe_rbf=True is not supported")
        if model.activation != "ssp":
            raise ValueError('Only activation="ssp" is supported')
        if model.neighbor_embedding:
            raise ValueError("neighbor_embedding=True is not supported")
        if model.cutoff_lower != 0.0:
            raise ValueError("Only lower_cutoff=0.0 is supported")
        if model.aggr != "add":
            raise ValueError('Only aggr="add" is supported')

        super().__init__()
        self.model = model

        self.neighbors = CFConvNeighbors(self.model.cutoff_upper)

        offset = self.model.distance_expansion.offset
        width = offset[1] - offset[0]
        self.convs = [
            CFConv(
                gaussianWidth=width,
                activation="ssp",
                weights1=inter.mlp[0].weight.T,
                biases1=inter.mlp[0].bias,
                weights2=inter.mlp[2].weight.T,
                biases2=inter.mlp[2].bias,
            )
            for inter in self.model.interactions
        ]

    def forward(self,
                z: pt.Tensor,
                pos: pt.Tensor,
                batch: pt.Tensor,
                q: Optional[pt.Tensor] = None,
                s: Optional[pt.Tensor] = None
                ) -> Tuple[pt.Tensor, Optional[pt.Tensor], pt.Tensor, pt.Tensor, pt.Tensor]:

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
        return "Optimized: " + repr(self.model)


def optimize(model):

    assert isinstance(model, TorchMD_Net)

    if isinstance(model.representation_model, TorchMD_GN):
        model.representation_model = TorchMD_GN_optimized(model.representation_model)
    else:
        raise ValueError("Unsupported model! Only TorchMD_GN is suppored.")

    return model
