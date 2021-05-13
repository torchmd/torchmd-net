from abc import abstractmethod, ABCMeta

import torch
from torch import nn
from torch.autograd import grad


class BaseWrapper(nn.Module, metaclass=ABCMeta):
    r"""Base class for model wrappers.

    Children of this class should implement the `forward` method,
    which calls `self.model(z, pos, batch=batch)` at some point.
    Wrappers that are applied before the output network should return
    the model's output, `z`, `pos` and `batch`. Wrappers that are applied
    after the output network should only return the model's output.
    """
    def __init__(self, model):
        super(BaseWrapper, self).__init__()
        self.model = model

    def reset_parameters(self):
        self.model.reset_parameters()

    @abstractmethod
    def forward(self, z, pos, batch=None):
        return
    

class Derivative(BaseWrapper):
    def forward(self, z, pos, batch=None):
        pos.requires_grad_(True)
        out = self.model(z, pos, batch=batch)
        dy = -grad(out, pos, grad_outputs=torch.ones_like(out),
                   create_graph=True, retain_graph=True)[0]
        return out, dy


class Standardize(BaseWrapper):
    def __init__(self, model, mean, std):
        super(Standardize, self).__init__(model)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, z, pos, batch=None):
        out = self.model(z, pos, batch=batch)
        if self.std is not None:
            out = out * self.std
        if self.mean is not None:
            out = out + self.mean
        return out


class Atomref(BaseWrapper):
    def __init__(self, model, atomref, max_z):
        super(Atomref, self).__init__(model)
        self.register_buffer('initial_atomref', atomref)
        self.atomref = nn.Embedding(max_z, 1)
        self.atomref.weight.data.copy_(atomref)

    def reset_parameters(self):
        super(Atomref, self).reset_parameters()
        self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, pos, batch=None):
        x, z, pos, batch = self.model(z, pos, batch=batch)
        x = x + self.atomref(z)
        return x, z, pos, batch


class AtomFilter(BaseWrapper):
    def __init__(self, model, min_z):
        super(AtomFilter, self).__init__(model)
        self.min_z = min_z

    def forward(self, z, pos, batch=None):
        x, z, pos, batch = self.model(z, pos, batch=batch)

        n_samples = len(batch.unique())

        # drop atoms according to the filter
        atom_mask = z > self.min_z
        x = x[atom_mask]
        z = z[atom_mask]
        pos = pos[atom_mask]
        batch = batch[atom_mask]
        
        assert len(batch.unique()) == n_samples,\
            ('Some samples were completely filtered out by the atom filter. '
             f'Make sure that at least one atom per sample exists with Z > {self.min_z}.')
        return x, z, pos, batch
