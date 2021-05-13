from torchmdnet.module import LNNP
from torchmdnet.models.torchmd_gn import TorchMD_GN
from torchmdnet.models.torchmd_t import TorchMD_T
from torchmdnet.models.wrappers import (Derivative, Standardize, Reduce,
                                        Atomref, OutputNetwork, AtomFilter)


def create_model(args, atomref=None, mean=None, std=None):
    r"""Nested model structure. Modules REDUCE, OUTPUT_NET and REPRESENTATION are required,
    the remaining modules may be included to achieve certain functionality.

        model = Derivative(Standardize(REDUCE(Atomref(OUTPUT_NET(AtomFilter(REPRESENTATION()))))))
                                |________________|
                                        |
                                mutually exclusive

        Derivative     - computes the derivative of the model's output (compute's force predictions)
        Standardize    - returns `pred * std + mean` where mean and std come from the dataset
        REDUCE         - aggregates atoms of one molecule (optionally computes dipole moment magnitude)
        Atomref        - adds atomic reference data to each atomic prediction, based on atom type `z`
        OUTPUT_NET     - transforms atomic embeddings into atomwise scalar predictions
        AtomFilter     - removes atomic embeddings where `atom_type <= remove_threshold`
        REPRESENTATION - computes featurization given the atomic neighborhood (e.g. Graph Network, Transformer)
    """

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

    # REPRESENTATION
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

    # AtomFilter
    if not args.derivative and args.atom_filter > -1:
        model = AtomFilter(model, args.atom_filter)
    elif args.atom_filter > -1:
        raise ValueError('Derivative and atom filter can\'t be used together')

    # OUTPUT_NET
    model = OutputNetwork(model, args.embedding_dimension, args.activation)

    # Atomref
    if atomref is not None and not args.dipole:
        model = Atomref(model, atomref, args.max_z)

    # REDUCE
    model = Reduce(model, args.reduce_op, args.dipole)

    # Standardize
    if (mean is not None or std is not None) and not args.dipole:
        model = Standardize(model, mean, std)

    # Derivative
    if args.derivative:
        model = Derivative(model)
    return model


def load_model(filepath, args=None, device='cpu'):
    if args is None:
        # use args from the checkpoint
        return LNNP.load_from_checkpoint(
            filepath, map_location=device
        ).model.to(device)
    else:
        # create model with new args and only load model state_dict
        model = create_model(args).to(device)
        
        state_dict = LNNP.load_from_checkpoint(
            filepath, map_location=device
        ).model.state_dict()

        model.load_state_dict(state_dict)
        return model