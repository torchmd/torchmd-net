from torchmdnet.models import TorchMD_GN, TorchMD_T

def create_model(args):
    model_args = dict(
        hidden_channels=args.embedding_dimension,
        num_rbf=args.num_rbf,
        rbf_type=args.rbf_type,
        trainable_rbf=args.trainable_rbf,
        activation=args.activation,
        neighbor_embedding=args.neighbor_embedding,
        cutoff_lower=args.cutoff_lower,
        cutoff_upper=args.cutoff_upper,
        derivative=args.derivative
    )

    if args.model == 'graph-network':
        return TorchMD_GN(
            num_filters=args.embedding_dimension,
            num_interactions=args.num_layers,
            **model_args
        )
    elif args.model == 'transformer':
        return TorchMD_T(
            num_layers=args.num_layers,
            attn_activation=args.attn_activation,
            num_heads=args.num_heads,
            distance_influence=args.distance_influence,
            **model_args
        )

    raise ValueError(f'Unknown architecture: {args.model}')
