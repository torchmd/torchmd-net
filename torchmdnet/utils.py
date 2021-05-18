import yaml
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from os.path import dirname, join, exists


def train_val_test_split(dset_len, val_ratio, test_ratio, seed, order=None):
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
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith('yaml') or values.name.endswith('yml'):
            with values as f:
                namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
        else:
            raise ValueError('configuration file must end with yaml or yml')


class LoadFromCheckpoint(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        hparams_path = join(dirname(values), 'hparams.yaml')
        if not exists(hparams_path):
            print('Failed to locate the checkpoint\'s hparams.yaml file. Relying on command line args.')
            return
        with open(hparams_path, 'r') as f:
            namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
        namespace.__dict__.update(load_model=values)


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
