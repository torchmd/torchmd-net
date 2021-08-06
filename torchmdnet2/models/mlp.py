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


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression of scalars
  '''
  def __init__(self, layer_widths: List[int] = [10, 10, 1],
                        activation_func: nn.Module = torch.nn.Tanh()):
    super(MLP, self).__init__()
    layers = []
    for w_in, w_out in zip(layer_widths[:-2], layer_widths[1:-1]):
        layers.append(nn.Linear(w_in, w_out, bias=True, dtype=torch.float64))
        layers.append(activation_func)
    # last layer without activation function and bias
    layers.append(nn.Linear(layer_widths[-2], layer_widths[-1], bias=False, dtype=torch.float64))

    self.layers = nn.Sequential(*layers)


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


class MLPModel(Model):
    '''A multi-layer perceptron (MLP) model using a representation of the atomic environment.

    Args:
        calculator (nn.Module): The representation class to use
        hidden_widths (list): List of the widths of the MLP's hidden layers. The input and output width are set by the calculator size end
        activation_func (nn.Module, optional): The activation function to use
            (default: :obj:`torch.nn.Tanh()`)
        property (str, optional): The name of the target property.
            (default: :obj:`"energy"`)
        derivative (str, optional): The name of the derivative of the target property.
            (default: :obj:`"forces"`)
        factors (dict, optional): weight factors used in the computation of the loss. For instance computing the loss only with forces, factors could be :obj:`{"forces": 1.}`.
            (default: :obj:`None`)
        loss (callable, optional): function to compute the loss which signature is :obj:`(prediction, data, factors)`
            (default: :obj:`None`)
        kwargs (optional): hypers for models.utils.Model
    '''
    def __init__(self, calculator: nn.Module,
                    hidden_widths: List[int],
                    activation_func: nn.Module = torch.nn.Tanh(),
                    property: str='energy',
                    derivative: str ='forces',
                    factors: Optional[Dict[str, float]]=None,
                    loss: Optional[Callable] = None,
                    **kwargs):
        super(MLPModel, self).__init__(**kwargs)
        self.save_hyperparameters()

        self.calculator = calculator
        layer_widths = [self.calculator.size()] + hidden_widths + [1]
        self.mlp = MLP(layer_widths, activation_func)

        self.property = property
        self.derivative = derivative

        if loss is None:
            self.loss = default_loss
        else:
            self.loss = loss

        if factors is None:
            self.factors = {self.derivative: 1., self.property: 1.}
        else:
            self.factors = factors


    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)

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
                create_graph=self.training
            )[0]
            return {self.property:y, self.derivative:dy_dr}
        else:
            return {self.property:y}
