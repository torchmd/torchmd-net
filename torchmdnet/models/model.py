import re
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn

from torchmdnet.models.torchmd_gn import TorchMD_GN
from torchmdnet.models.torchmd_t import TorchMD_T
from torchmdnet.models.torchmd_et import TorchMD_ET
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet import priors


def create_model(args, prior_model=None, mean=None, std=None):
    if prior_model is not None and (mean is not None or std is not None):
        rank_zero_warn('Prior model and standardize are given, only using the prior model.')

    shared_args = dict(
        hidden_channels=args['embedding_dimension'],
        num_layers=args['num_layers'],
        num_rbf=args['num_rbf'],
        rbf_type=args['rbf_type'],
        trainable_rbf=args['trainable_rbf'],
        activation=args['activation'],
        neighbor_embedding=args['neighbor_embedding'],
        cutoff_lower=args['cutoff_lower'],
        cutoff_upper=args['cutoff_upper'],
        max_z=args['max_z'],
        max_num_neighbors=args['max_num_neighbors'],
    )

    # representation network
    if args['model'] == 'graph-network':
        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args['embedding_dimension'],
            **shared_args
        )
    elif args['model'] == 'transformer':
        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args['attn_activation'],
            num_heads=args['num_heads'],
            distance_influence=args['distance_influence'],
            **shared_args
        )
    elif args['model'] == 'equivariant-transformer':
        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args['attn_activation'],
            num_heads=args['num_heads'],
            distance_influence=args['distance_influence'],
            **shared_args
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # atom filter
    if not args['derivative'] and args['atom_filter'] > -1:
        representation_model = AtomFilter(representation_model, args['atom_filter'])
    elif args['atom_filter'] > -1:
        raise ValueError('Derivative and atom filter can\'t be used together')

    # prior model
    if args['prior_model'] and prior_model is None:
        assert 'prior_args' in args, (f'Requested prior model {args.prior_model} but the '
                                      f'arguments are lacking the key "prior_args".')
        assert hasattr(priors, args['prior_model']), (f'Unknown prior model {args["prior_model"]}. '
                                                      f'Available models are {", ".join(priors.__all__)}')
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = getattr(priors, args['prior_model'])(**args['prior_args'])

    # create output network
    output_model = getattr(output_modules, args['output_model'])(is_equivariant, args['embedding_dimension'], args['activation'])

    # combine representation and output network
    model = TorchMD_Net(representation_model, output_model, prior_model=prior_model,
                                       reduce_op=args['reduce_op'], mean=mean, std=std, derivative=args['derivative'])
    return model


def load_model(filepath, args=None, device='cpu'):
    ckpt = torch.load(filepath, map_location='cpu')
    if args is None:
        args = ckpt['hyper_parameters']

    model = create_model(args)

    state_dict = {re.sub(r'^model\.', '', k): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model.to(device)


class TorchMD_Net(nn.Module):
    def __init__(self, representation_model, output_model, prior_model=None,
                 reduce_op='add', mean=None, std=None, derivative=False):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        if output_model.allow_prior_model:
            self.prior_model = prior_model
        else:
            self.prior_model = None
            rank_zero_warn(('Prior model was given but the output model does '
                            'not allow prior models. Dropping the prior model.'))

        self.reduce_op = reduce_op
        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer('mean', mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer('std', std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        representation = self.representation_model(z, pos, batch=batch)

        if len(representation) == 5:
            x, v, z, pos, batch = representation
        else:
            v = None
            x, z, pos, batch = representation

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # apply prior model
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        # standardize if no prior model is given and the output model allows priors
        if self.prior_model is None and self.output_model.allow_prior_model:
            if self.std is not None:
                out = out * self.std
            if self.mean is not None:
                out = out + self.mean

        out = self.output_model.post_reduce(out)

        # compute gradients with respect to coordinates
        if self.derivative:
            dy = -grad(out, pos, grad_outputs=torch.ones_like(out),
                    create_graph=True, retain_graph=True)[0]
            return out, dy
        return out
