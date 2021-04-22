import sys
import os
import torch
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

try:
    from pytorch_lightning.plugins import DDPPlugin
except ImportError:
    # compatibility for PyTorch Lightning versions < 1.2.0
    from pytorch_lightning.plugins.ddp_plugin import DDPPlugin

from torchmdnet.utils import LoadFromFile, save_argparse
from torchmdnet import LNNP


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile)# keep first
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-patience', type=int, default=10, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='Stop training after this many epochs without improvement')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='/tmp/logs', help='log file')
    parser.add_argument('--load-model', default=None, help='Restart training using a model checkpoint')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--val-ratio', type=float, default=0.05, help='Percentage of validation set')
    parser.add_argument('--test-ratio', type=float, default=0.10922, help='Percentage of test set')
    parser.add_argument('--test-interval', type=int, default=10, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--redirect', type=bool, default=False, help='Redirect stdout and stderr to log_dir/log')

    # dataset specific
    parser.add_argument('--dataset', default=None, type=str, choices=['QM9', 'ANI1', 'CG'], help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='~/data', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--coords', default=None, type=str, help='CG coordinate files glob')
    parser.add_argument('--forces', default=None, type=str, help='CG force files glob')
    parser.add_argument('--embed', default=None, type=str, help='CG embedding files glob')

    # model architecture
    parser.add_argument('--model', type=str, default='graph-network', choices=['graph-network', 'transformer'], help='Which model to train')

    # architectural args
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of interaction layers in the model')
    parser.add_argument('--num-rbf', type=int, default=64, help='Number of radial basis functions in model')
    parser.add_argument('--activation', type=str, default='silu', choices=['silu', 'ssp', 'tanh', 'sigmoid'], help='Activation function')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=['gauss', 'expnorm'], help='Type of distance expansion')
    parser.add_argument('--trainable-rbf', type=bool, default=False, help='If distance expansion functions should be trainable')
    parser.add_argument('--neighbor-embedding', type=bool, default=False, help='If a neighbor embedding should be applied before interactions')

    # Transformer specific
    parser.add_argument('--distance-influence', type=str, default='both', choices=['keys', 'values', 'both'], help='Where distance information is included inside the attention')
    parser.add_argument('--attn-activation', default='silu', choices=['silu', 'ssp', 'tanh', 'sigmoid'], help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')

    # other args
    parser.add_argument('--label', default=None, type=str, help='Target property, e.g. energy_U0, forces')
    parser.add_argument('--derivative', default=False, type=bool, help='If true, take the derivative of the prediction w.r.t coordinates')
    parser.add_argument('--cutoff-lower', type=float, default=0.0, help='Lower cutoff in model')
    parser.add_argument('--cutoff-upper', type=float, default=5.0, help='Upper cutoff in model')
    # fmt: on
 
    args = parser.parse_args()

    assert args.label is not None, 'Please specify a label.'

    if args.redirect:
        sys.stdout = open(os.path.join(args.log_dir, 'log'), 'w')
        sys.stderr = sys.stdout
 
    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    save_argparse(args, os.path.join(args.log_dir, 'input.yaml'), exclude=['conf'])

    return args


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = LNNP(args)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor='val_loss',
        save_top_k=10, # -1 to save all
        period=args.save_interval,
        filename='{epoch}-{val_loss:.4f}-{test_loss:.4f}'
    )
    early_stopping = EarlyStopping('val_loss', patience=args.early_stopping_patience)
    
    tb_logger = pl.loggers.TensorBoardLogger(args.log_dir, name='tensorbord', version='')
    csv_logger = pl.loggers.CSVLogger(args.log_dir, name='', version='')

    ddp_plugin = None
    if 'ddp' in args.distributed_backend:
        ddp_plugin = DDPPlugin(find_unused_parameters=False)

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        gpus=args.ngpus,
        num_nodes=args.num_nodes,
        distributed_backend=args.distributed_backend,
        default_root_dir=args.log_dir,
        auto_lr_find=False,
        resume_from_checkpoint=args.load_model,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping],
        logger=[tb_logger, csv_logger],
        reload_dataloaders_every_epoch=False,
        enable_pl_optimizer=True,
        precision=args.precision,
        plugins=[ddp_plugin]
    )

    trainer.fit(model)

    # run test set after completing the fit
    trainer.test()


if __name__ == "__main__":
    main()
