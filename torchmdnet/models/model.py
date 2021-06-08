import re
import torch
from pytorch_lightning.utilities import rank_zero_warn

from torchmdnet.models.torchmd_gn import TorchMD_GN
from torchmdnet.models.torchmd_t import TorchMD_T
from torchmdnet.models.output_modules import OutputNetwork
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

    # REPRESENTATION
    if args['model'] == 'graph-network':
        model = TorchMD_GN(
            num_filters=args['embedding_dimension'],
            **shared_args
        )
    elif args['model'] == 'transformer':
        model = TorchMD_T(
            attn_activation=args['attn_activation'],
            num_heads=args['num_heads'],
            distance_influence=args['distance_influence'],
            **shared_args
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # AtomFilter
    if not args['derivative'] and args['atom_filter'] > -1:
        model = AtomFilter(model, args['atom_filter'])
    elif args['atom_filter'] > -1:
        raise ValueError('Derivative and atom filter can\'t be used together')

    if args['prior_model'] and prior_model is None:
        assert 'prior_args' in args, (f'Requested prior model {args.prior_model} but the '
                                      f'arguments are lacking the key "prior_args".')
        assert hasattr(priors, args['prior_model']), (f'Unknown prior model {args["prior_model"]}. '
                                                      f'Available models are {", ".join(priors.__all__)}')
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = getattr(priors, args['prior_model'])(**args['prior_args'])

    # OUTPUT NET
    model = OutputNetwork(model, args['embedding_dimension'], args['activation'],
                          reduce_op=args['reduce_op'], dipole=args['dipole'],
                          prior_model=prior_model, mean=mean, std=std,
                          derivative=args['derivative'])
    return model


def load_model(filepath, args=None, device='cpu'):
    ckpt = torch.load(filepath, map_location='cpu')
    if args is None:
        args = ckpt['hyper_parameters']

    model = create_model(args)

    state_dict = {re.sub(r'^model\.', '', k): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model.to(device)
