from mkl import get_max_threads,set_num_threads
set_num_threads(16)

import sys, os
import numpy as np
import pyemma as pe
import pyemma
import torch  # pytorch
import matplotlib.pyplot as plt

sys.path.insert(0,'/home/musil/git/torchmd-net/')
from torchmdnet2.dataset import ChignolinDataset, DataModule
from torchmdnet2.models import LNNP, SchNet, MLPModel, CGnet
from torchmdnet2.utils import LoadFromFile, save_argparse
from torchmdnet2.simulation import Simulation

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pytorch_lightning.plugins import DDPPlugin

from torch_geometric.data import DataLoader

from torch.nn import Embedding, Sequential, Linear, ModuleList


def plot_tica(baseline_model, dataset, lag=10, tica=None):
    # compute distances all of the beads
    baseline_model.cpu()  # moves the tensors onto the cpu
    if isinstance(dataset, np.ndarray):  # check if the dataset is an numpy array
        n_traj, n_samp, n_beads, _ = traj.shape
        features = []
        for i_traj in range(n_traj):
            _ = baseline_model.geom_feature(torch.from_numpy(traj[i_traj]))
            feat = baseline_model.geom_feature.distances
            features.append(feat)
    else:
        _ = baseline_model.geom_feature(dataset.data.pos.reshape((-1, baseline_model.n_beads, 3)))
        feat = baseline_model.geom_feature.distances

        if 'traj_idx' in dataset.data:
            traj_ids = dataset.data.traj_idx
            n_traj = np.unique(traj_ids).shape[0]
            traj_strides = np.cumsum([0]+(np.bincount(traj_ids)).tolist(), dtype=int)

            features = []
            for i_traj in range(n_traj):
                st, nd = traj_strides[i_traj], traj_strides[i_traj+1]
                features.append(feat[st:nd].numpy())
        else:
            features = feat.numpy()

    if tica is None:
        tica = pe.coordinates.tica(features, lag=lag, dim=2)
        tica_concatenated = np.concatenate(tica.get_output())
    else:
        Xproj = tica.transform(features)
        tica_concatenated = np.concatenate(Xproj)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    pyemma.plots.plot_feature_histograms(
        tica_concatenated, ['IC {}'.format(i + 1) for i in range(tica.dimension())], ax=axes[0])
    pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=axes[1], cbar=False, logscale=True)
    pyemma.plots.plot_free_energy(*tica_concatenated[:, :2].T, ax=axes[2], legacy=False)
    for ax in axes.flat[1:]:
        ax.set_xlabel('IC 1')
        ax.set_ylabel('IC 2')
    fig.tight_layout()
    return fig, axes, tica


if __name__ == "__main__":
    device = torch.device('cuda')

    n_sims = 1000
    n_timesteps = 10000
    save_interval = 10

    chignolin_dataset = ChignolinDataset('/local_scratch/musil/datasets/chignolin/')

    baseline_model = chignolin_dataset.get_baseline_model(n_beads=10)  # doesn't work without specifying n_beads

    model = MLPModel.load_from_checkpoint("/local_scratch/musil/chign/epoch=5-validation_loss=27.2908-test_loss=0.0000.ckpt")


    ids = np.arange(0, len(chignolin_dataset),len(chignolin_dataset)//n_sims).tolist()
    init = chignolin_dataset[ids]
    initial_coords = torch.cat([init[i].pos.reshape((1,-1,3)) for i in range(len(init))], dim=0).to(device=device)
    initial_coords.requires_grad_()

    sim_embeddings = torch.cat([init[i].z.reshape((1,-1)) for i in range(len(init))], dim=0).to(device=device)

    # initializing the Net with the learned weights and biases and preparing it for evaluation
    chignolin_net = CGnet(model, baseline_model).eval().to(device=device)


    sim = Simulation(chignolin_net, initial_coords, sim_embeddings, length=n_timesteps,
                    save_interval=save_interval, beta=baseline_model.beta,
                    save_potential=True, device=device,
                    log_interval=100, log_type='print',
                    batch_size=100)

    traj = sim.simulate()

    torch.save(traj, '/local_scratch/musil/chign/traj.pt')

    fig,_, tica = plot_tica(baseline_model, chignolin_dataset, lag=10)
    plt.savefig('/local_scratch/musil/chign/ref_traj.png', dpi=300, bbox_inches='tight')

    fig,_,_ = plot_tica(baseline_model, traj, tica=tica)
    plt.savefig('/local_scratch/musil/chign/simulated_traj.png', dpi=300, bbox_inches='tight')