# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import sys
import os
import yaml
import argparse
import logging
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from torchmdnet.module import LNNP
from torchmdnet import datasets, priors, models
from torchmdnet.data import DataModule
from torchmdnet.loss import loss_class_mapping
from torchmdnet.models import output_modules
from torchmdnet.models.model import create_prior_models
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping, dtype_mapping
from torchmdnet.utils import (
    LoadFromFile,
    LoadFromCheckpoint,
    save_argparse,
    number,
    check_logs,
)
from lightning_utilities.core.rank_zero import rank_zero_warn


def get_argparse():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-patience', type=int, default=10, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-metric', type=str, default='val', choices=['train', 'val'], help='Metric to monitor when deciding whether to reduce learning rate')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Factor by which to multiply the learning rate when the metric stops improving')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='Stop training after this many epochs without improvement')
    parser.add_argument('--early-stopping-monitor', type=str, default='val_total_mse_loss', choices=['train_total_mse_loss', 'val_total_mse_loss'], help='Metric to monitor for early stopping')
    parser.add_argument('--reset-trainer', type=bool, default=False, help='Reset training metrics (e.g. early stopping, lr) when loading a model checkpoint')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--ema-alpha-y', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of y. Must be between 0 and 1.')
    parser.add_argument('--ema-alpha-neg-dy', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of dy. Must be between 0 and 1.')
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of GPU nodes for distributed training with the Lightning Trainer.')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32, 64], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='/tmp/logs', help='log file')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=None, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=0.05, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=0.1, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--test-interval', type=int, default=-1, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--redirect', type=bool, default=False, help='Redirect stdout and stderr to log_dir/log')
    parser.add_argument('--gradient-clipping', type=float, default=0.0, help='Gradient clipping norm')
    parser.add_argument('--remove-ref-energy', action='store_true', help='If true, remove the reference energy from the dataset for delta-learning. Total energy can still be predicted by the model during inference by turning this flag off when loading.  The dataset must be compatible with Atomref for this to be used.')
    parser.add_argument('--checkpoint-monitor', type=str, default='val_total_mse_loss', choices=['train_total_mse_loss', 'val_total_mse_loss'], help='Metric to monitor for writing out best models')
    parser.add_argument('--load-weights', default=None, type=str, help='Load the weights of an existing model')
    # dataset specific
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='~/data', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset-arg', default=None, help='Additional dataset arguments. Needs to be a dictionary.')
    parser.add_argument('--coord-files', default=None, type=str, help='Custom coordinate files glob')
    parser.add_argument('--embed-files', default=None, type=str, help='Custom embedding files glob')
    parser.add_argument('--energy-files', default=None, type=str, help='Custom energy files glob')
    parser.add_argument('--force-files', default=None, type=str, help='Custom force files glob')
    parser.add_argument('--dataset-preload-limit', default=1024, type=int, help='Custom and HDF5 datasets will preload to RAM datasets that are less than this size in MB')
    parser.add_argument('--y-weight', default=1.0, type=float, help='Weighting factor for y label in the loss function')
    parser.add_argument('--neg-dy-weight', default=1.0, type=float, help='Weighting factor for neg_dy label in the loss function')
    parser.add_argument('--train-loss', default='mse_loss', type=str, choices=loss_class_mapping.keys(), help='Loss function to use during training')
    parser.add_argument('--train-loss-arg', default=None, help='Additional arguments for the loss function. Needs to be a dictionary.')

    # model architecture
    parser.add_argument('--model', type=str, default='graph-network', choices=models.__all_models__, help='Which model to train')
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')
    parser.add_argument('--output-mlp-num-layers', type=int, default=0, help='If the output model uses an MLP this will be the number of hidden layers, excluding the input and output layers.')
    parser.add_argument('--prior-model', type=str, default=None, help='Which prior model to use. It can be a string, a dict if you want to add arguments for it or a dicts to add more than one prior. e.g. {"Atomref": {"max_z":100}, "Coulomb":{"max_num_neighs"=100, "lower_switch_distance"=4, "upper_switch_distance"=8}', action="extend", nargs="*")

    # architectural args
    parser.add_argument('--charge', type=bool, default=False, help='Model needs a total charge. Set this to True if your dataset contains charges and you want them passed down to the model.')
    parser.add_argument('--spin', type=bool, default=False, help='Model needs a spin state. Set this to True if your dataset contains spin states and you want them passed down to the model.')
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of interaction layers in the model')
    parser.add_argument('--num-rbf', type=int, default=64, help='Number of radial basis functions in model')
    parser.add_argument('--activation', type=str, default='silu', choices=list(act_class_mapping.keys()), help='Activation function')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=list(rbf_class_mapping.keys()), help='Type of distance expansion')
    parser.add_argument('--trainable-rbf', type=bool, default=False, help='If distance expansion functions should be trainable')
    parser.add_argument('--neighbor-embedding', type=bool, default=False, help='If a neighbor embedding should be applied before interactions')
    parser.add_argument('--aggr', type=str, default='add', help='Aggregation operation for CFConv filter output. Must be one of \'add\', \'mean\', or \'max\'')

    # Transformer specific
    parser.add_argument('--distance-influence', type=str, default='both', choices=['keys', 'values', 'both', 'none'], help='Where distance information is included inside the attention')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()), help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')

    # Equivariant Transformer specific
    parser.add_argument('--vector-cutoff', type=bool, default=False, help='If true, the vector features are weighted by the cutoff function during message passing, forcing the energy to be continuous at the cutoff.')

    # TensorNet specific
    parser.add_argument('--equivariance-invariance-group', type=str, default='O(3)', help='Equivariance and invariance group of TensorNet')
    parser.add_argument('--box-vecs', type=lambda x: list(yaml.safe_load(x)), default=None, help="""Box vectors for periodic boundary conditions. The vectors `a`, `b`, and `c` represent a triclinic box and must satisfy
        certain requirements:
        `a[1] = a[2] = b[2] = 0`;`a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff`;`a[0] >= 2*b[0]`;`a[0] >= 2*c[0]`;`b[1] >= 2*c[1]`;
        These requirements correspond to a particular rotation of the system and reduced form of the vectors, as well as the requirement that the cutoff be no larger than half the box width.
    Example: [[1,0,0],[0,1,0],[0,0,1]]""")
    parser.add_argument('--static_shapes', type=bool, default=False, help='If true, TensorNet will use statically shaped tensors for the network, making it capturable into a CUDA graphs. In some situations static shapes can lead to a speedup, but it increases memory usage.')

    # other args
    parser.add_argument('--check_errors', type=bool, default=True, help='Will check if max_num_neighbors is not enough to contain all neighbors. This is incompatible with CUDA graphs.')
    parser.add_argument('--derivative', default=False, type=bool, help='If true, take the derivative of the prediction w.r.t coordinates')
    parser.add_argument('--cutoff-lower', type=float, default=0.0, help='Lower cutoff in model')
    parser.add_argument('--cutoff-upper', type=float, default=5.0, help='Upper cutoff in model')
    parser.add_argument('--atom-filter', type=int, default=-1, help='Only sum over atoms with Z > atom_filter')
    parser.add_argument('--max-z', type=int, default=100, help='Maximum atomic number that fits in the embedding matrix')
    parser.add_argument('--max-num-neighbors', type=int, default=32, help='Maximum number of neighbors to consider in the network')
    parser.add_argument('--standardize', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'], help='Reduce operation to apply to atomic predictions')
    parser.add_argument('--wandb-use', default=False, type=bool, help='Defines if wandb is used or not')
    parser.add_argument('--wandb-name', default='training', type=str, help='Give a name to your wandb run')
    parser.add_argument('--wandb-project', default='training_', type=str, help='Define what wandb Project to log to')
    parser.add_argument('--wandb-resume-from-id', default=None, type=str, help='Resume a wandb run from a given run id. The id can be retrieved from the wandb dashboard')
    parser.add_argument('--tensorboard-use', default=False, type=bool, help='Defines if tensor board is used or not')

    # fmt: on
    return parser


def get_args():
    parser = get_argparse()
    args = parser.parse_args()
    if args.redirect:
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    os.makedirs(os.path.abspath(args.log_dir), exist_ok=True)
    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def fix_state_dict(ckpt):
    import re

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    # In ET, before we had output_model.output_network.{0,1}.update_net.[0-9].{weight,bias}
    # Now we have output_model.output_network.{0,1}.update_net.layers.[0-9].{weight,bias}
    # In other models, we had output_model.output_network.{0,1}.{weight,bias},
    # which is now output_model.output_network.layers.{0,1}.{weight,bias}
    # This change was introduced in https://github.com/torchmd/torchmd-net/pull/314
    patterns = [
        (
            r"output_model.output_network.(\d+).update_net.(\d+).",
            r"output_model.output_network.\1.update_net.layers.\2.",
        ),
        (
            r"output_model.output_network.([02]).(weight|bias)",
            r"output_model.output_network.layers.\1.\2",
        ),
    ]
    for p in patterns:
        state_dict = {re.sub(p[0], p[1], k): v for k, v in state_dict.items()}
    return state_dict


def main():
    args = get_args()
    if args.remove_ref_energy:
        if args.prior_model is None:
            args.prior_model = []
        if not isinstance(args.prior_model, list):
            args.prior_model = [args.prior_model]
        args.prior_model.append({"Atomref": {"enable": False}})

    pl.seed_everything(args.seed, workers=True)

    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.setup("fit")

    prior_models = create_prior_models(vars(args), data.dataset)
    args.prior_args = [p.get_init_args() for p in prior_models]
    # initialize lightning module
    model = LNNP(args, prior_model=prior_models, mean=data.mean, std=data.std)
    if args.load_weights is not None:
        print(f"Loading weights from {args.load_weights}")
        ckpt = torch.load(args.load_weights, map_location="cpu")
        model.model.load_state_dict(fix_state_dict(ckpt))

    mon = args.checkpoint_monitor
    chkpoint_name = f"epoch={{epoch}}-{mon.split('_')[0]}_loss={{{mon}:.4f}}"
    if len(data.idx_test):
        chkpoint_name += "-test_loss={test_total_l1_loss:.4f}"

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor=args.checkpoint_monitor,
        save_top_k=10,  # -1 to save all
        every_n_epochs=args.save_interval,
        filename=chkpoint_name,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    if args.early_stopping_monitor is not None:
        early_stopping = EarlyStopping(
            args.early_stopping_monitor, patience=args.early_stopping_patience
        )
        callbacks.append(early_stopping)

    check_logs(args.log_dir)
    csv_logger = CSVLogger(args.log_dir, name="", version="")
    _logger = [csv_logger]

    if args.wandb_use:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            save_dir=args.log_dir,
            resume="must" if args.wandb_resume_from_id is not None else None,
            id=args.wandb_resume_from_id,
        )
        _logger.append(wandb_logger)

    if args.tensorboard_use:
        tb_logger = TensorBoardLogger(
            args.log_dir, name="tensorboard", version="", default_hp_metric=False
        )
        _logger.append(tb_logger)
    if args.test_interval > 0:
        rank_zero_warn(
            f"WARNING: Test set will be evaluated every {args.test_interval} epochs. This will slow down training."
        )

    trainer = pl.Trainer(
        strategy="auto",
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=args.ngpus,
        num_nodes=args.num_nodes,
        default_root_dir=args.log_dir,
        callbacks=callbacks,
        logger=_logger,
        precision=args.precision,
        gradient_clip_val=args.gradient_clipping,
        inference_mode=False,
        # Test-during-training requires reloading the dataloaders every epoch
        reload_dataloaders_every_n_epochs=1 if args.test_interval > 0 else 0,
    )

    trainer.fit(model, data, ckpt_path=None if args.reset_trainer else args.load_model)

    # run test set after completing the fit
    model = LNNP.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer = pl.Trainer(
        logger=_logger,
        inference_mode=False,
        accelerator="auto",
        devices=args.ngpus,
        num_nodes=args.num_nodes,
    )
    trainer.test(model, data)


if __name__ == "__main__":
    main()
