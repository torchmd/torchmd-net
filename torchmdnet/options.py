from torchmdnet.utils import LoadFromCheckpoint, number
from torchmdnet import datasets, priors, models
from torchmdnet.models import output_modules
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping

import argparse
# Initialize an empty dictionary called options
options = {}

# Populate the options dictionary with argument properties
options['load-model'] = {'type': None, 'default': None, 'choices': None, 'help': 'Restart training using a model checkpoint', 'action': LoadFromCheckpoint}
options['conf'] = {'abbrev': 'c', 'type': open, 'default': None, 'choices': None, 'help': 'Configuration yaml file'}
options['num-epochs'] = {'type': int, 'default': 300, 'choices': None, 'help': 'number of epochs'}
options['batch-size'] = {'type': int, 'default': 32, 'choices': None, 'help': 'batch size'}
options['inference-batch-size'] = {'type': int, 'default': None, 'choices': None, 'help': 'Batchsize for validation and tests.'}
options['lr'] = {'type': float, 'default': '1e-4', 'choices': None, 'help': 'learning rate'}
options['lr-patience'] = {'type': int, 'default': 10, 'choices': None, 'help': 'Patience for lr-schedule. Patience per eval-interval of validation'}
options['lr-metric'] = {'type': str, 'default': 'val_total_mse_loss', 'choices': ['train_total_mse_loss', 'val_total_mse_loss'], 'help': 'Metric to monitor when deciding whether to reduce learning rate'}
options['lr-min'] = {'type': float, 'default': '1e-6', 'choices': None, 'help': 'Minimum learning rate before early stop'}
options['lr-factor'] = {'type': float, 'default': 0.8, 'choices': None, 'help': 'Factor by which to multiply the learning rate when the metric stops improving'}
options['lr-warmup-steps'] = {'type': int, 'default': 0, 'choices': None, 'help': 'How many steps to warm-up over. Defaults to 0 for no warm-up'}
options['early-stopping-patience'] = {'type': int, 'default': 30, 'choices': None, 'help': 'Stop training after this many epochs without improvement'}
options['reset-trainer'] = {'type': bool, 'default': False, 'choices': None, 'help': 'Reset training metrics (e.g. early stopping, lr) when loading a model checkpoint'}
options['weight-decay'] = {'type': float, 'default': 0.0, 'choices': None, 'help': 'Weight decay strength'}
options['ema-alpha-y'] = {'type': float, 'default': 1.0, 'choices': None, 'help': 'The amount of influence of new losses on the exponential moving average of y'}
options['ema-alpha-neg-dy'] = {'type': float, 'default': 1.0, 'choices': None, 'help': 'The amount of influence of new losses on the exponential moving average of dy'}
options['ngpus'] = {'type': int, 'default': -1, 'choices': None, 'help': 'Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus'}
options['num-nodes'] = {'type': int, 'default': 1, 'choices': None, 'help': 'Number of nodes'}
options['precision'] = {'type': int, 'default': 32, 'choices': [16, 32, 64], 'help': 'Floating point precision'}
options['log-dir'] = {'abbrev':'l', 'type': None, 'default': '/tmp/logs', 'choices': None, 'help': 'log file'}
options['splits'] = {'type': None, 'default': None, 'choices': None, 'help': 'Npz with splits idx_train, idx_val, idx_test'}
options["train-size"] = {'type': number, 'default': None, 'help': 'Percentage/number of samples in training set (None to use all remaining samples)'}
options["val-size"] = {'type': number, 'default': 0.05, 'help': 'Percentage/number of samples in validation set (None to use all remaining samples)'}
options["test-size"] = {'type': number, 'default': 0.1, 'help': 'Percentage/number of samples in test set (None to use all remaining samples)'}
options["test-interval"] = {'type': int, 'default': -1, 'help': 'Test interval, one test per n epochs (default: 10)'}
options["save-interval"] = {'type': int, 'default': 10, 'help': 'Save interval, one save per n epochs (default: 10)'}
options["seed"] = {'type': int, 'default': 1, 'help': 'random seed (default: 1)'}
options["num-workers"] = {'type': int, 'default': 4, 'help': 'Number of workers for data prefetch'}
options["redirect"] = {'type': bool, 'default': False, 'help': 'Redirect stdout and stderr to log_dir/log'}
options["gradient-clipping"] = {'type': float, 'default': 0.0, 'help': 'Gradient clipping norm'}
options["dataset"] = {'type': str, 'default': None, 'choices': datasets.__all__, 'help': 'Name of the torch_geometric dataset'}
options["dataset-root"] = {'type': str, 'default': '~/data', 'help': 'Data storage directory (not used if dataset is "CG")'}
options["dataset-arg"] = {'type': str, 'default': None, 'help': 'Additional dataset arguments, e.g. target property for QM9 or molecule for MD17. Need to be specified in JSON format i.e. \'{"molecules": "aspirin,benzene"}\''}
options["coord-files"] = {'type': str, 'default': None, 'help': 'Custom coordinate files glob'}
options["embed-files"] = {'type': str, 'default': None, 'help': 'Custom embedding files glob'}
options["energy-files"] = {'type': str, 'default': None, 'help': 'Custom energy files glob'}
options["force-files"] = {'type': str, 'default': None, 'help': 'Custom force files glob'}
options["y-weight"] = {'type': float, 'default': 1.0, 'help': 'Weighting factor for y label in the loss function'}
options["neg-dy-weight"] = {'type': float, 'default': 1.0, 'help': 'Weighting factor for neg_dy label in the loss function'}
# Model architecture options

options["model"] = {'type': str, 'default': 'graph-network', 'choices': models.__all__, 'help': 'Which model to train'}

options["output-model"] = {'type': str, 'default': 'Scalar', 'choices': output_modules.__all__, 'help': 'The type of output model'}

options["prior-model"] = {'type': str, 'default': None, 'choices': priors.__all__, 'help': 'Which prior model to use'}



# Architectural args
options["charge"] = {'type': bool, 'default': False, 'help': 'Model needs a total charge. Set this to True if your dataset contains charges and you want them passed down to the model.'}
options["spin"] = {'type': bool, 'default': False, 'help': 'Model needs a spin state. Set this to True if your dataset contains spin states and you want them passed down to the model.'}
options["embedding-dimension"] = {'type': int, 'default': 256, 'help': 'Embedding dimension'}
options["num-layers"] = {'type': int, 'default': 6, 'help': 'Number of interaction layers in the model'}
options["num-rbf"] = {'type': int, 'default': 64, 'help': 'Number of radial basis functions in model'}
options["activation"] = {'type': str, 'default': 'silu', 'choices': list(act_class_mapping.keys()), 'help': 'Activation function'}
options["rbf-type"] = {'type': str, 'default': 'expnorm', 'choices': list(rbf_class_mapping.keys()), 'help': 'Type of distance expansion'}
options["trainable-rbf"] = {'type': bool, 'default': False, 'help': 'If distance expansion functions should be trainable'}
options["neighbor-embedding"] = {'type': bool, 'default': False, 'help': 'If a neighbor embedding should be applied before interactions'}
options["aggr"] = {'type': str, 'default': 'add', 'help': "Aggregation operation for CFConv filter output. Must be one of 'add', 'mean', or 'max'"}

# Transformer specific
options["distance-influence"] = {'type': str, 'default': 'both', 'choices': ['keys', 'values', 'both', 'none'], 'help': 'Where distance information is included inside the attention'}
options["attn-activation"] = {'type': str, 'default': 'silu', 'choices': list(act_class_mapping.keys()), 'help': 'Attention activation function'}
options["num-heads"] = {'type': int, 'default': 8, 'help': 'Number of attention heads'}


# TensorNet specific

options["equivariance-invariance-group"] = {'type': str, 'default': 'O(3)', 'help': 'Equivariance and invariance group of TensorNet'}


# Other args

options["derivative"] = {'type': bool, 'default': False, 'help': 'If true, take the derivative of the prediction w.r.t coordinates'}
options["cutoff-lower"] = {'type': float, 'default': 0.0, 'help': 'Lower cutoff in model'}
options["cutoff-upper"] = {'type': float, 'default': 5.0, 'help': 'Upper cutoff in model'}
options["atom-filter"] = {'type': int, 'default': -1, 'help': 'Only sum over atoms with Z > atom_filter'}
options["max-z"] = {'type': int, 'default': 100, 'help': 'Maximum atomic number that fits in the embedding matrix'}
options["max-num-neighbors"] = {'type': int, 'default': 32, 'help': 'Maximum number of neighbors to consider in the network'}
options["standardize"] = {'type': bool, 'default': False, 'help': 'If true, multiply prediction by dataset std and add mean'}
options["reduce-op"] = {'type': str, 'default': 'add', 'choices': ['add', 'mean'], 'help': 'Reduce operation to apply to atomic predictions'}
options["wandb-use"] = {'type': bool, 'default': False, 'help': 'Defines if wandb is used or not'}
options["wandb-name"] = {'type': str, 'default': 'training', 'help': 'Give a name to your wandb run'}
options["wandb-project"] = {'type': str, 'default': 'training_', 'help': 'Define what wandb Project to log to'}
options["wandb-resume-from-id"] = {'type': str, 'default': None, 'help': 'Resume a wandb run from a given run id. The id can be retrieved from the wandb dashboard'}
options["tensorboard-use"] = {'type': bool, 'default': False, 'help': 'Defines if tensor board is used or not'}
def get_argparse():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    # Dynamically add arguments from the options dictionary
    for opt_name, opt_props in options.items():
        args = ['--' + opt_name]
        if 'abbrev' in opt_props:
            args.append('-' + opt_props['abbrev'])
        kwargs = {k: v for k, v in opt_props.items() if k != 'abbrev' and v is not None}
        parser.add_argument(*args, **kwargs)
    return parser
