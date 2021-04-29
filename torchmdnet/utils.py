import yaml
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

try:
    from pytorch_lightning.trainer.states import RunningStage
except ImportError:
    # compatibility for PyTorch lightning versions < 1.2.0
    RunningStage = None


def train_val_test_split(dset_len,val_ratio,test_ratio, seed, order=None):
    shuffle = True if order is None else False
    valtest_ratio = val_ratio + test_ratio
    idx_train = list(range(dset_len))
    idx_test = []
    idx_val = []
    if valtest_ratio > 0 and dset_len > 0:
        idx_train, idx_tmp = train_test_split(range(dset_len), test_size=valtest_ratio, random_state=seed, shuffle=shuffle)
        if test_ratio == 0:
            idx_val = idx_tmp
        elif val_ratio == 0:
            idx_test = idx_tmp
        else:
            test_val_ratio = test_ratio / (test_ratio + val_ratio)
            idx_val, idx_test = train_test_split(idx_tmp, test_size=test_val_ratio, random_state=seed, shuffle=shuffle)

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(dataset_len, val_ratio, test_ratio, seed, filename=None, splits=None, order=None):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits['idx_train']
        idx_val = splits['idx_val']
        idx_test = splits['idx_test']
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, val_ratio, test_ratio, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return torch.from_numpy(idx_train), torch.from_numpy(idx_val), torch.from_numpy(idx_test)


class LoadFromFile(argparse.Action):
    #parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__ (self, parser, namespace, values, option_string=None):
        if values.name.endswith('yaml') or values.name.endswith('yml'):
            with values as f:
                namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
        else:
            raise ValueError('configuration file must end with yaml or yml')


def save_argparse(args, filename, exclude=None):
    if filename.endswith('yaml') or filename.endswith('yml'):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, 'w'))
    else:
        raise ValueError('Configuration file should end with yaml or yml')


class TestingContext:
    def __init__(self, lightning_module):
        self.lightning_module = lightning_module
        self.version_1_3 = hasattr(TrainerState, 'TESTING')

    def __enter__(self):
        if RunningStage is None:
            # PyTorch Lightning < 1.2.0
            self.lightning_module.trainer.testing = True
        elif not self.version_1_3:
            # PyTorch Lightning >= 1.2.0 and < 1.3.0
            self._stage = self.lightning_module.trainer._running_stage
            self.lightning_module.trainer._set_running_stage(RunningStage.TESTING, self)
        else:
            # PyTorch Lightning >= 1.3.0
            self._stage = self.lightning_module.trainer._running_stage
            self.lightning_module.trainer._running_stage = RunningStage.TESTING
            self._state = self.lightning_module.trainer.state
            self.lightning_module.trainer.state = TrainerState.TESTING

    def __exit__(self, type, value, traceback):
        if RunningStage is None:
            # PyTorch Lightning < 1.2.0
            self.lightning_module.trainer.testing = False
        elif not self.version_1_3:
            # PyTorch Lightning >= 1.2.0 and < 1.3.0
            self.lightning_module.trainer._set_running_stage(self._stage, self)
        else:
            # PyTorch Lightning >= 1.3.0
            self.lightning_module.trainer._running_stage = self._stage
            self.lightning_module.trainer.state = self._state


class TrainCSVLogger(CSVLogger):
    r"""
    Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from torchmdnet.utils import TrainCSVLogger
        >>> logger = TrainCSVLogger("logs", name="my_exp_name", reqiures_metric="train_loss")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        requires_metric: A string or list of strings which are required in the metrics dict
            in order to save it to the CSV.
        prefix: A string to put at the beginning of metric keys.
    """

    def __init__(self, *args, requires_metric=None, **kwargs):
        if isinstance(requires_metric, str):
            requires_metric = [requires_metric]
        self.requires_metric = requires_metric
        super(TrainCSVLogger, self).__init__(*args, **kwargs)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if self.requires_metric and len([key for key in self.requires_metric if key not in metrics]) == 0:
            super(TrainCSVLogger, self).log_metrics(metrics, step)
