# torchmd-net

## Installation
Create a new conda environment using Python 3.8 via
```
conda create --name torchmd python=3.8
conda activate torchmd
```

Then, install PyTorch according to your hardware specifications (more information [here](https://pytorch.org/get-started/locally/#start-locally)), e.g. for CUDA 11.1 and the most recent version of PyTorch use
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

Download and install the `torchmd-net` repository via
```
git clone https://github.com/compsciencelab/torchmd-net.git
pip install -e torchmd-net/
```

Finally, install `torch-geometric` with its dependencies as it is specified [here](https://github.com/rusty1s/pytorch_geometric#installation). Example for PyTorch 1.8 and CUDA 11.1:
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
```

## Usage
Specifying training arguments can either be done via a configuration yaml file or through command line arguments directly. An example configuration file for a TorchMD Graph Network can be found at [examples/graph-network.yaml](https://github.com/compsciencelab/torchmd-net/blob/main/examples/graph-network.yaml). For an example on how to train the network on the QM9 dataset, see [examples/train_GN_QM9.sh](https://github.com/compsciencelab/torchmd-net/blob/main/examples/train_GN_QM9.sh). GPUs can be selected by their index by listing the device IDs (coming from `nvidia-smi`) in the `CUDA_VISIBLE_DEVICES` environment variable. Otherwise, the argument `--ngpus` can be used to select the number of GPUs to train on (-1 uses all available GPUs or the ones specified in `CUDA_VISIBLE_DEVICES`).
```
mkdir output
CUDA_VISIBLE_DEVICES=0 python torchmd-net/scripts/torchmd_train.py --conf torchmd-net/examples/graph-network.yaml --dataset QM9 --log-dir output/
```

## Multi-Node Training
__Currently does not work with the most recent PyTorch Lightning version. Tested up to pytorch-lightning==1.2.10__

In order to train models on multiple nodes some environment variables have to be set, which provide all necessary information to PyTorch Lightning. In the following we provide an example bash script to start training on two machines with two GPUs each. The script has to be started once on each node. Once [`train.py`](https://github.com/compsciencelab/torchmd-net/blob/main/scripts/train.py) is started on all nodes, a network connection between the nodes will be established using NCCL.

In addition to the environment variables the argument `--num-nodes` has to be specified with the number of nodes involved during training.

```
export NODE_RANK=0
export MASTER_ADDR=hostname1
export MASTER_PORT=12910

mkdir -p output
CUDA_VISIBLE_DEVICES=0,1 python torchmd-net/scripts/train.py --conf torchmd-net/examples/graph-network.yaml --num-nodes 2 --log-dir output/
```

- `NODE_RANK` : Integer indicating the node index. Must be `0` for the main node and incremented by one for each additional node.
- `MASTER_ADDR` : Hostname or IP address of the main node. The same for all involved nodes.
- `MASTER_PORT` : A free network port for communication between nodes. PyTorch Lightning suggests port `12910` as a default.

### Known Limitations
- Due to the way PyTorch Lightning calculates the number of required DDP processes, all nodes must use the same number of GPUs. Otherwise training will not start or crash.
- We observe a 50x decrease in performance when mixing nodes with different GPU architectures (tested with RTX 2080 Ti and RTX 3090).
