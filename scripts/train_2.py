import sys
import os
import torch
import argparse
import yaml
import subprocess

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


sys.path.insert(0,'/local/git/torchmd-net2/')

from torchmdnet2.utils import LoadFromFile, save_argparse, Args
from torchmdnet2.models import SchNet
from torchmdnet2.dataset import DataModule

from pytorch_lightning.utilities.cli import LightningCLI


if __name__ == "__main__":

    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    cli = LightningCLI(SchNet, DataModule)
