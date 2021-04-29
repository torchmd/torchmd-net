import sys
import os
import torch
import argparse
import yaml
import subprocess

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

try:
    from pytorch_lightning.plugins import DDPPlugin
except ImportError:
    # compatibility for PyTorch Lightning versions < 1.2.0
    from pytorch_lightning.plugins.ddp_plugin import DDPPlugin

# sys.path.insert(0,'../')

from torchmdnet2.utils import LoadFromFile, save_argparse, Args
from torchmdnet2.models import SchNet
from torchmdnet2.dataset import DataModule

from pytorch_lightning.utilities.cli import LightningCLI


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--conf', '-c', type=str, help='filename for the configuration file. only yaml files are supported')

    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    if args.redirect:
        sys.stdout = open(os.path.join(conf['log_dir'], 'log'), 'w')
        sys.stderr = sys.stdout

    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }
    conf['git'] = git

    save_argparse(conf, os.path.join(conf['log_dir'], 'input.yaml'))

    return conf


def main():
    conf = get_args()
    args = Args(**conf)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args_model = conf['model']

    assert args_model['name'] in model_map, f'Name of the model should be one of {list(model_map.keys())}'

    Network = model_map[args_model.pop('name')]

    network = Network(**args_model)
    model = LNNP(network)

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

    cli = LightningCLI(SchNet, DataModule)
