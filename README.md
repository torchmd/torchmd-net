# torchmd-net

## Installation

```
git clone https://github.com/compsciencelab/torchmd-net.git
pip install torchmd-net/
```

## Usage
Specifying training arguments can either be done via a configuration `.yaml` file or through command line arguments directly. An example configuration file for training a TorchMD Graph Network on QM9 can be found at [examples/graph-network.yaml](https://github.com/compsciencelab/torchmd-net/blob/main/examples/graph-network.yaml). GPUs can be selected by listing the desired device IDs (coming from `nvidia-smi`) in the `CUDA_VISIBLE_DEVICES` environment variable.
```
mkdir output
CUDA_VISIBLE_DEVICES=0 python torchmd-net/scripts/torchmd_train.py --conf torchmd-net/examples/graph-network.yaml --log-dir output/
```

## Multi-Node Training
__Currently does not work with the most recent PyTorch Lightning version. Tested for pytorch-lightning==1.1.0__

In order to train models on multiple nodes some environment variables have to be set, which provide all necessary information to PyTorch Lightning. In the following we provide an example bash script to start training on two machines with two GPUs each. The script has to be started once on each node. Once [`train.py`](https://github.com/compsciencelab/torchmd-net/blob/main/scripts/train.py) is started on all nodes, a network connection between the nodes will be established using NCCL.

```
export NODE_RANK=0
export MASTER_ADDR=hostname1
export MASTER_PORT=12910

mkdir -p output
CUDA_VISIBLE_DEVICES=0,1 python torchmd-net/scripts/train.py --conf torchmd-net/examples/graph-network.yaml --log-dir output/ --num-nodes 2
```

- `NODE_RANK` : Integer indicating the node index. Must be `0` for the main node and incremented by one for each additional node.
- `MASTER_ADDR` : Hostname or IP address of the main node. The same for all involved nodes.
- `MASTER_PORT` : A free network port for communication between nodes. PyTorch Lightning suggests port `12910` as a default.

### Known Limitations
- Due to the way PyTorch Lightning calculates the number of required DDP processes, all nodes must use the same number of GPUs. Otherwise training will not start or crash.
- We observe a 50x decrease in performance when mixing nodes with different GPU architectures (tested with RTX 2080 Ti and RTX 3090).
