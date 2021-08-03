import re
from typing import Optional, List
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter
from copy import deepcopy
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.models.torchmd_gn import TorchMD_GN
from torchmdnet.models.torchmd_t import TorchMD_T
from torchmdnet.models.torchmd_et import TorchMD_ET
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet import priors


def create_model(args, prior_model=None, mean=None, std=None):
    if prior_model is not None and (mean is not None or std is not None):
        rank_zero_warn(
            "Prior model and standardize are given, only using the prior model."
        )

    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
    )

    # representation network
    if args["model"] == "graph-network":
        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"], **shared_args
        )
    elif args["model"] == "transformer":
        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformer":
        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # atom filter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")

    # prior model
    if args["prior_model"] and prior_model is None:
        assert "prior_args" in args, (
            f"Requested prior model {args.prior_model} but the "
            f'arguments are lacking the key "prior_args".'
        )
        assert hasattr(priors, args["prior_model"]), (
            f'Unknown prior model {args["prior_model"]}. '
            f'Available models are {", ".join(priors.__all__)}'
        )
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])

    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
    )

    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
        output_heads=args["output_heads"] if "output_heads" in args else ["default"],
    )
    return model


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        assert key in args, "Unknown hyperparameter '{key}'."
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model.to(device)


class TorchMD_Net(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_heads=["default"],
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model

        if output_model.allow_prior_model:
            self.prior_model = prior_model
        else:
            self.prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )

        self.head_map = {head: i for i, head in enumerate(output_heads)}
        self.output_models = nn.ModuleList(
            [deepcopy(output_model) for _ in range(len(output_heads))]
        )

        self.reduce_op = reduce_op
        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        for model in self.output_models:
            model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(
        self,
        z,
        pos,
        batch: Optional[torch.Tensor] = None,
        head_labels: Optional[torch.Tensor] = None,
    ):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)

        # apply the output networks
        res = []
        for outnet in self.output_models:
            res.append(outnet.pre_reduce(x, v, z, pos, batch))
        x = torch.hstack(res)

        # If the head of each sample is defined (i.e. which head it matches to), select only that prediction
        if head_labels is not None and not any([head_labels is None]):
            label_idx = torch.tensor([self.head_map[lt] for lt in head_labels])
            idx = torch.zeros_like(batch)
            idx[batch] = label_idx[batch]
            x = x.gather(1, idx.long().view(-1, 1))

        # apply prior model
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        # standardize if no prior model is given and the output model allows priors
        if self.prior_model is None and self.output_models[0].allow_prior_model:
            if self.std is not None:
                out = out * self.std
            if self.mean is not None:
                out = out + self.mean

        out = self.output_models[0].post_reduce(out)

        # compute gradients with respect to coordinates
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
