# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import argparse
import lightning.pytorch as pl
from torchmdnet.module import LNNP
from torchmdnet import datasets, priors, models
from torchmdnet.data import DataModule
from torchmdnet.models import output_modules
from torchmdnet.utils import LoadFromFile, number


def get_argparse():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', type=str, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--inference-batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32, 64], help='Floating point precision')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')

    # dataset specific
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='~/data', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset-arg', default=None, type=str, help='Additional dataset arguments, e.g. target property for QM9 or molecule for MD17. Need to be specified in JSON format i.e. \'{"molecules": "aspirin,benzene"}\'')
    parser.add_argument('--coord-files', default=None, type=str, help='Custom coordinate files glob')
    parser.add_argument('--embed-files', default=None, type=str, help='Custom embedding files glob')
    parser.add_argument('--energy-files', default=None, type=str, help='Custom energy files glob')
    parser.add_argument('--force-files', default=None, type=str, help='Custom force files glob')
    parser.add_argument('--y-weight', default=1.0, type=float, help='Weighting factor for y label in the loss function')
    parser.add_argument('--neg-dy-weight', default=1.0, type=float, help='Weighting factor for neg_dy label in the loss function')

    # model architecture
    parser.add_argument('--model', type=str, default='graph-network', choices=models.__all_models__, help='Which model to train')
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')
    parser.add_argument('--prior-model', type=str, default=None, choices=priors.__all__, help='Which prior model to use')

    parser.add_argument('--train-size', type=number, default=0, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=0, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=11, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-dir', '-l', default='/tmp/logs', help='log file')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--standardize', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')

    # fmt: on
    return parser


def main():
    parser = get_argparse()
    args = parser.parse_args()

    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.setup("fit")

    # run test set after completing the fit
    model = LNNP.load_from_checkpoint(args.load_model)
    trainer = pl.Trainer(
        logger=False,
        inference_mode=False,
        accelerator="auto",
        devices=1,
        num_nodes=1,
    )
    trainer.test(model, data)


if __name__ == "__main__":
    main()
