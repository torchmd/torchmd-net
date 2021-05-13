import torchmdnet
from torchmdnet.models import TorchMD_GN, TorchMD_T, OutputNetwork
from torchmdnet.models.wrappers import AtomFilter, Atomref, Standardize, Derivative


def create_model(args, atomref=None, mean=None, std=None):
    assert atomref is None or (mean is None and std is None),\
        'Use either atomref or standardization, both is not supported.'

    shared_args = dict(
        hidden_channels=args.embedding_dimension,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        rbf_type=args.rbf_type,
        trainable_rbf=args.trainable_rbf,
        activation=args.activation,
        neighbor_embedding=args.neighbor_embedding,
        cutoff_lower=args.cutoff_lower,
        cutoff_upper=args.cutoff_upper,
        max_z=args.max_z
    )

    # representation model
    if args.model == 'graph-network':
        model = TorchMD_GN(
            num_filters=args.embedding_dimension,
            **shared_args
        )
    elif args.model == 'transformer':
        model = TorchMD_T(
            attn_activation=args.attn_activation,
            num_heads=args.num_heads,
            distance_influence=args.distance_influence,
            **shared_args
        )
    else:
        raise ValueError(f'Unknown architecture: {args.model}')

    # atom filter
    if not args.derivative and args.atom_filter > -1:
        model = AtomFilter(model, args.atom_filter)
    elif args.atom_filter > -1:
        raise ValueError('Derivative and atom filter can\'t be used together')

    # atomref
    if atomref is not None and not args.dipole:
        model = Atomref(model, atomref, args.max_z)

    # output network
    model = OutputNetwork(model, args.embedding_dimension, args.activation,
                          readout=args.readout, dipole=args.dipole)

    # standardize
    if (mean is not None or std is not None) and not args.dipole:
        model = Standardize(model, mean, std)

    # derivative
    if args.derivative:
        model = Derivative(model)

    model.reset_parameters()
    return model


def load_model(filepath, args=None, device='cpu'):
    if args is None:
        # use args from the checkpoint
        return torchmdnet.LNNP.load_from_checkpoint(filepath, map_location=device).model.to(device)
    else:
        # create model with new args and only load model state_dict
        model = create_model(args).to(device)
        state_dict = torchmdnet.LNNP.load_from_checkpoint(filepath, map_location=device).model.state_dict()
        model.load_state_dict(state_dict)
        return model