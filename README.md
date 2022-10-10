# TorchMD-NET

TorchMD-NET provides state-of-the-art graph neural networks and equivariant transformer neural networks potentials for learning molecular potentials. It offers an efficient and fast implementation and it is integrated in GPU-accelerated molecular dynamics code like [ACEMD](https://www.acellera.com/products/molecular-dynamics-software-gpu-acemd/) and [OpenMM](https://www.openmm.org). See the full paper at https://arxiv.org/abs/2202.02541.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/torchmd/torchmd-net.git
    cd torchmd-net
    ```

2. Install Mambaforge (https://github.com/conda-forge/miniforge/#mambaforge). It is recommended to use `mamba` rather than `conda`. `conda` is known to produce broken enviroments with PyTorch.

3. Create an environment and activate it:
    ```
    mamba env create -f environment.yml
    mamba activate torchmd-net
    ```

4. Install TorchMD-NET into the environment:
    ```
    pip install -e .
    ```

## Cite
```
@inproceedings{
tholke2021equivariant,
title={Equivariant Transformers for Neural Network based Molecular Potentials},
author={Philipp Th{\"o}lke and Gianni De Fabritiis},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=zNHzqZ9wrRB}
}
```


## Usage
Specifying training arguments can either be done via a configuration yaml file or through command line arguments directly. An example configuration file for a TorchMD Graph Network can be found in [examples/](https://github.com/compsciencelab/torchmd-net/blob/main/examples). For an example on how to train the network on the QM9 dataset, see [examples/](https://github.com/compsciencelab/torchmd-net/blob/main/examples). GPUs can be selected by their index by listing the device IDs (coming from `nvidia-smi`) in the `CUDA_VISIBLE_DEVICES` environment variable. Otherwise, the argument `--ngpus` can be used to select the number of GPUs to train on (-1 uses all available GPUs or the ones specified in `CUDA_VISIBLE_DEVICES`).
```
mkdir output
CUDA_VISIBLE_DEVICES=0 torchmd-train --conf torchmd-net/examples/ET-QM9.yaml --log-dir output/
```

## Pretrained models
Pretrained models are available at https://github.com/torchmd/torchmd-net/tree/main/examples.

## Creating a new dataset
If you want to train on custom data, first have a look at `torchmdnet.datasets.Custom`, which provides functionalities for 
loading a NumPy dataset consisting of atom types and coordinates, as well as energies, forces or both as the labels.
Alternatively, you can implement a custom class according to the torch-geometric way of implementing a dataset. That is, 
derive the `Dataset` or `InMemoryDataset` class and implement the necessary functions (more info [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-your-own-datasets)). The dataset must return torch-geometric `Data` 
objects, containing at least the keys `z` (atom types) and `pos` (atomic coordinates), as well as `y` (label), `dy` (derivative of the label w.r.t atom coordinates) or both.

### Custom prior models
In addition to implementing a custom dataset class, it is also possible to add a custom prior model to the model. This can be
done by implementing a new prior model class in `torchmdnet.priors` and adding the argument `--prior-model <PriorModelName>`.
As an example, have a look at `torchmdnet.priors.Atomref`.

## Multi-Node Training

In order to train models on multiple nodes some environment variables have to be set, which provide all necessary information to PyTorch Lightning. In the following we provide an example bash script to start training on two machines with two GPUs each. The script has to be started once on each node. Once `torchmd-train` is started on all nodes, a network connection between the nodes will be established using NCCL.

In addition to the environment variables the argument `--num-nodes` has to be specified with the number of nodes involved during training.

```
export NODE_RANK=0
export MASTER_ADDR=hostname1
export MASTER_PORT=12910

mkdir -p output
CUDA_VISIBLE_DEVICES=0,1 torchmd-train --conf torchmd-net/examples/ET-QM9.yaml.yaml --num-nodes 2 --log-dir output/
```

- `NODE_RANK` : Integer indicating the node index. Must be `0` for the main node and incremented by one for each additional node.
- `MASTER_ADDR` : Hostname or IP address of the main node. The same for all involved nodes.
- `MASTER_PORT` : A free network port for communication between nodes. PyTorch Lightning suggests port `12910` as a default.

### Known Limitations
- Due to the way PyTorch Lightning calculates the number of required DDP processes, all nodes must use the same number of GPUs. Otherwise training will not start or crash.
- We observe a 50x decrease in performance when mixing nodes with different GPU architectures (tested with RTX 2080 Ti and RTX 3090).
