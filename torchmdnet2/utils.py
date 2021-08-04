import yaml
import argparse
import numpy as np
import torch
import inspect
from sklearn.model_selection import train_test_split

try:
    from pytorch_lightning.trainer.states import RunningStage
except ImportError:
    # compatibility for PyTorch lightning versions < 1.2.0
    RunningStage = None

def is_notebook():
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def load_yaml(fn):
    with open(fn, 'r') as f:
        data = yaml.safe_load(f)
    return data


def train_val_test_split(dset_len,val_ratio,test_ratio, seed=None, order=None):
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


def make_splits(dataset_len, val_ratio, test_ratio, seed=None, filename=None, splits=None, order=None):
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


from torch_geometric.data.data import size_repr

from argparse import Namespace

class Args(Namespace):
    def __init__(self,**kwargs):
        for key, item in kwargs.items():
            if isinstance(item, dict):
                self[key] = Args(**item)
            else:
                self[key] = item

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return '{}({})'.format(cls, ', '.join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return '{}(\n{}\n)'.format(cls, ',\n'.join(info))


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

    def __enter__(self):
        if RunningStage is None:
            # PyTorch Lightning < 1.2.0
            self.lightning_module.trainer.testing = True
        else:
            # PyTorch Lightning >= 1.2.0
            self._stage = self.lightning_module.running_stage
            self.lightning_module.trainer._set_running_stage(RunningStage.TESTING, self)

    def __exit__(self, type, value, traceback):
        if RunningStage is None:
            # PyTorch Lightning < 1.2.0
            self.lightning_module.trainer.testing = False
        else:
            # PyTorch Lightning >= 1.2.0
            self.lightning_module.trainer._set_running_stage(self._stage, self)
