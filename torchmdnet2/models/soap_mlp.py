from torch_scatter import scatter
import torch
from torch import nn
from typing import Dict, Optional, Callable, Union, List

from .utils import Model
from .schnet import ShiftedSoftplus

def default_loss(pred, data, facs) -> torch.Tensor:
    losses = []
    for k in facs.keys():
        ll = facs[k] * (pred[k] - data[k]).pow(2).mean()
        losses.append(ll)
    loss = torch.stack(losses).sum()
    return loss



# act_class_mapping = {
#     'ssp': ShiftedSoftplus,
#     'silu': torch.nn.SiLU,
#     'tanh': torch.nn.Tanh,
#     'signmoid': torch.nn.Sigmoid
# }


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression of scalars
  '''
  def __init__(self, layer_widths: List[int] = [10, 10, 1],
                        activation_func: nn.Module = torch.nn.Tanh()):
    super(MLP).__init__()
    layers = []
    for w_in, w_out in zip(layer_widths[:-1], layer_widths[1:]):
        layers.append(nn.Linear(w_in, w_out, bias=True))
        layers.append(activation_func)
    # last layer to predict scalars
    layers.append(nn.Linear(layer_widths[-1], 1, bias=False))

    self.layers = nn.Sequential(*layers)


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


class MLPModel(Model):
    def __init__(self, calculator: nn.Module,
                    mlp_widths: List[int],
                    activation_func: nn.Module = torch.nn.Tanh(),
                    property: str='energy',
                    derivative: str ='forces',
                    factors: Optional[Dict[str, float]]=None,
                    loss: Optional[Callable] = None):
        super(MLPModel, self).__init__()
        self.save_hyperparameters()

        self.calculator = calculator
        layer_widths = [self.calculator.size()] + mlp_widths
        self.mlp = MLP(layer_widths, activation_func)
        self.property = property
        self.derivative = derivative
        if loss is None:
            self.loss = default_loss
        if factors is None:
            self.factors = {self.derivative: 1., self.property: 1.}

    def step(self, data, stage):
        with torch.set_grad_enabled(self.derivative is not None):
            pred = self(data)

        loss = self.loss(pred, data, self.factors)
        # Add sync_dist=True to sync logging across all GPU workers
        self.log(f'{stage}_loss', loss.sqrt(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss


    def forward(self, data):
        if self.derivative is not None:
            data.pos.requires_grad_(True)

        features = self.calculator(data)
        yi = self.mlp(features)

        y = scatter(yi, data.batch, dim=0)

        if self.derivative is not None:
            dy_dr = -torch.autograd.grad(
                y,
                data.pos,
                grad_outputs=torch.ones_like(y),
                retain_graph=True,
            )[0]
            return {self.property:y, self.derivative:dy_dr}
        else:
            return {self.property:y}
