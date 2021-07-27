import ase
from torchmdnet.models.utils import act_class_mapping
from torch_scatter import scatter
from typing import Optional, List
import torch
from torch import nn
from torch.autograd import grad


class OutputNetwork(nn.Module):
    def __init__(
        self,
        representation_model,
        hidden_channels,
        activation="silu",
        reduce_op="add",
        dipole=False,
        prior_model=None,
        mean=None,
        std=None,
        derivative=False,
        heads=["default"],
    ):
        super(OutputNetwork, self).__init__()
        self.representation_model = representation_model

        self.hidden_channels = hidden_channels
        self.activation = activation
        self.reduce_op = reduce_op
        self.dipole = dipole
        self.prior_model = prior_model
        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        std = torch.scalar_tensor(1) if std is None else std

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

        act_class = act_class_mapping[activation]

        self.head_map = {head: i for i, head in enumerate(heads)}
        self.output_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    act_class(),
                    nn.Linear(hidden_channels // 2, 1),
                )
                for _ in range(len(heads))
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

        for outnet in self.output_networks:
            nn.init.xavier_uniform_(outnet[0].weight)
            outnet[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(outnet[2].weight)
            outnet[2].bias.data.fill_(0)

    def forward(self, z, pos, label_types=None, batch: Optional[torch.Tensor] = None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        x, z, pos, batch = self.representation_model(z, pos, batch=batch)

        res = []
        for outnet in self.output_networks:
            res.append(outnet(x))
        x = torch.hstack(res)

        # If the type of each label is defined (i.e. which head it matches to), return only that prediction
        if label_types is not None and not any([label_types is None]):
            label_idx = torch.tensor([self.head_map[lt] for lt in label_types])
            idx = torch.zeros_like(batch)
            idx[batch] = label_idx[batch]
            x = x.gather(1, idx.long().view(-1, 1))

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            x = x * (pos - c[batch])
        elif self.prior_model is not None:
            # apply prior model
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        if not self.dipole and self.prior_model is None:
            if self.std is not None:
                out = out * self.std
            if self.mean is not None:
                out = out + self.mean

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, -dy
        # TODO: return only `out` once Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return out, None
