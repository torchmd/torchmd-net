import re
from typing import Optional, List, Tuple, Dict
import torch
from torch.autograd import grad
from torch import nn, Tensor
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet import priors
import warnings


def create_model(args, prior_model=None, mean=None, std=None):
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
        from torchmdnet.models.torchmd_gn import TorchMD_GN

        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"], aggr=args["aggr"], **shared_args
        )
    elif args["model"] == "transformer":
        from torchmdnet.models.torchmd_t import TorchMD_T

        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformer":
        from torchmdnet.models.torchmd_et import TorchMD_ET

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
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = create_prior_models(args)

    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"],
        activation=args["activation"],
        reduce_op=args["reduce_op"],
    )

    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        prior_model=prior_model,
        mean=mean,
        std=std,
        derivative=args["derivative"],
    )
    return model


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    # The following are for backward compatibility with models created when atomref was
    # the only supported prior.
    if 'prior_model.initial_atomref' in state_dict:
        state_dict['prior_model.0.initial_atomref'] = state_dict['prior_model.initial_atomref']
        del state_dict['prior_model.initial_atomref']
    if 'prior_model.atomref.weight' in state_dict:
        state_dict['prior_model.0.atomref.weight'] = state_dict['prior_model.atomref.weight']
        del state_dict['prior_model.atomref.weight']
    model.load_state_dict(state_dict)
    return model.to(device)


def create_prior_models(args, dataset=None):
    """Parse the prior_model configuration option and create the prior models."""
    prior_models = []
    if args['prior_model']:
        prior_model = args['prior_model']
        prior_names = []
        prior_args = []
        if not isinstance(prior_model, list):
            prior_model = [prior_model]
        for prior in prior_model:
            if isinstance(prior, dict):
                for key, value in prior.items():
                    prior_names.append(key)
                    if value is None:
                        prior_args.append({})
                    else:
                        prior_args.append(value)
            else:
                prior_names.append(prior)
                prior_args.append({})
        if 'prior_args' in args:
            prior_args = args['prior_args']
            if not isinstance(prior_args, list):
                prior_args = [prior_args]
        for name, arg in zip(prior_names, prior_args):
            assert hasattr(priors, name), (
                f"Unknown prior model {name}. "
                f"Available models are {', '.join(priors.__all__)}"
            )
            # initialize the prior model
            prior_models.append(getattr(priors, name)(dataset=dataset, **arg))
    return prior_models


class TorchMD_Net(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        mean=None,
        std=None,
        derivative=False,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        if not output_model.allow_prior_model and prior_model is not None:
            prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )
        if isinstance(prior_model, priors.base.BasePrior):
            prior_model = [prior_model]
        self.prior_model = None if prior_model is None else torch.nn.ModuleList(prior_model)

        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            for prior in self.prior_model:
                prior.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:

        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, pos, batch, q=q, s=s)

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply atom-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                x = prior.pre_reduce(x, z, pos, batch, extra_args)

        # aggregate atoms
        x = self.output_model.reduce(x, batch)

        # shift by data mean
        if self.mean is not None:
            x = x + self.mean

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        # apply molecular-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                y = prior.post_reduce(y, z, pos, batch, extra_args)

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
            dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return y, -dy
        # TODO: return only `out` once Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return y, None
