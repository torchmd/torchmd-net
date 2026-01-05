[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/torchmd/torchmd-net/actions/workflows/workflow.yml/badge.svg)](https://github.com/torchmd/torchmd-net/actions/workflows/workflow.yml)
[![Documentation Status](https://readthedocs.org/projects/torchmd-net/badge/?version=latest)](https://torchmd-net.readthedocs.io/en/latest/?badge=latest)  

# TorchMD-NET

TorchMD-NET provides state-of-the-art neural networks potentials (NNPs) and a mechanism to train them. It offers efficient and fast implementations if several NNPs and it is integrated in GPU-accelerated molecular dynamics code like [ACEMD](https://www.acellera.com/products/molecular-dynamics-software-gpu-acemd/), [OpenMM](https://www.openmm.org) and [TorchMD](https://github.com/torchmd/torchmd). TorchMD-NET exposes its NNPs as [PyTorch](https://pytorch.org) modules.


## Documentation

Documentation is available at https://torchmd-net.readthedocs.io  

## Available architectures

- [Equivariant Transformer (ET)](https://arxiv.org/abs/2202.02541)
- [Transformer (T)](https://arxiv.org/abs/2202.02541)
- [Graph Neural Network (GN)](https://arxiv.org/abs/2212.07492)
- [TensorNet](https://arxiv.org/abs/2306.06482)



## Installation  
TorchMD-Net is available as a pip package as well as in [conda-forge](https://conda-forge.org/)

As TorchMD-Net depends on PyTorch we need to add an additional index URL to the installation command as per [pytorch](https://pytorch.org/get-started/locally/)

```sh
# The following will install TorchMD-Net with PyTorch CUDA 12.6 version
ACCELERATOR=cu126 pip install torchmd-net --extra-index-url https://download.pytorch.org/whl/${ACCELERATOR}
```
where `ACCELERATOR` can be replaced with any of the available options in the above pytorch instructions
(i.e. `cu118`, `cu126`, `cu128`, `cu130`, `cpu`) depending on which version of pytorch should be installed.
Keep in mind tha the `cpu` versions are orders of magnitude slower than CUDA builds and should only be used 
for testing and not actual evaluation.

Alternatively it can be installed with conda or mamba with one of the following commands.
We recommend using [Miniforge](https://github.com/conda-forge/miniforge/) instead of anaconda.

```shell
# The following will install TorchMD-Net with PyTorch CUDA 12.6 version
CUDA=12.6 mamba install torchmd-net cuda-version=${CUDA} -c conda-forge
```
Again here CUDA can be replaced with any version supported by the current pytorch version.

When installing with conda it will not automatically install Triton which is used for speeding out
neighbor calculations as it only exist on conda for Linux. If you are running on Linux and want the full
performance of torchmd-net also run `mamba install triton -c conda-forge`

### Install from source  

TorchMD-Net is installed using pip with `pip install -e .` to create an editable install.  

## Usage
Specifying training arguments can either be done via a configuration yaml file or through command line arguments directly. Several examples of architectural and training specifications for some models and datasets can be found in [examples/](https://github.com/torchmd/torchmd-net/tree/main/examples). Note that if a parameter is present both in the yaml file and the command line, the command line version takes precedence.
GPUs can be selected by setting the `CUDA_VISIBLE_DEVICES` environment variable. Otherwise, the argument `--ngpus` can be used to select the number of GPUs to train on (-1, the default, uses all available GPUs or the ones specified in `CUDA_VISIBLE_DEVICES`). Keep in mind that the [GPU ID reported by nvidia-smi might not be the same as the one `CUDA_VISIBLE_DEVICES` uses](https://stackoverflow.com/questions/26123252/inconsistency-of-ids-between-nvidia-smi-l-and-cudevicegetname).  
For example, to train the Equivariant Transformer on the QM9 dataset with the architectural and training hyperparameters described in the paper, one can run:
```shell
mkdir output
CUDA_VISIBLE_DEVICES=0 torchmd-train --conf torchmd-net/examples/ET-QM9.yaml --log-dir output/
``` 
Run `torchmd-train --help` to see all available options and their descriptions.

## Inference with Pretrained models

### AceFF
Trained [AceFF models](https://huggingface.co/collections/Acellera/aceff-machine-learning-potentials) can be loaded and used for inference.
please see [here](https://github.com/torchmd/torchmd-net/tree/main/examples/aceff_examples) 


To load your own trained models see [here](https://github.com/torchmd/torchmd-net/tree/main/examples#loading-checkpoints) for instructions on how to load pretrained models.

## Creating a new dataset
If you want to train on custom data, first have a look at `torchmdnet.datasets.Custom`, which provides functionalities for 
loading a NumPy dataset consisting of atom types and coordinates, as well as energies, forces or both as the labels.
Alternatively, you can implement a custom class according to the torch-geometric way of implementing a dataset. That is, 
derive the `Dataset` or `InMemoryDataset` class and implement the necessary functions (more info [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-your-own-datasets)). The dataset must return torch-geometric `Data` 
objects, containing at least the keys `z` (atom types) and `pos` (atomic coordinates), as well as `y` (label), `neg_dy` (negative derivative of the label w.r.t atom coordinates) or both.

### Custom prior models
In addition to implementing a custom dataset class, it is also possible to add a custom prior model to the model. This can be
done by implementing a new prior model class in `torchmdnet.priors` and adding the argument `--prior-model <PriorModelName>`.
As an example, have a look at `torchmdnet.priors.Atomref`.

## Multi-Node Training

In order to train models on multiple nodes some environment variables have to be set, which provide all necessary information to PyTorch Lightning. In the following we provide an example bash script to start training on two machines with two GPUs each. The script has to be started once on each node. Once `torchmd-train` is started on all nodes, a network connection between the nodes will be established using NCCL.

In addition to the environment variables the argument `--num-nodes` has to be specified with the number of nodes involved during training.

```shell
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
- Some CUDA systems might hang during a multi-GPU parallel training. Try `export NCCL_P2P_DISABLE=1`, which disables direct peer to peer GPU communication.


## Cite
If you use TorchMD-NET in your research, please cite the following papers:

#### Main reference

```
@misc{pelaez2024torchmdnet,
title={TorchMD-Net 2.0: Fast Neural Network Potentials for Molecular Simulations}, 
author={Raul P. Pelaez and Guillem Simeon and Raimondas Galvelis and Antonio Mirarchi and Peter Eastman and Stefan Doerr and Philipp Thölke and Thomas E. Markland and Gianni De Fabritiis},
year={2024},
eprint={2402.17660},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```

#### TensorNet

```
@inproceedings{simeon2023tensornet,
title={TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials},
author={Guillem Simeon and Gianni De Fabritiis},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=BEHlPdBZ2e}
}
```

#### Equivariant Transformer
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

#### Graph Network 

```
@article{Majewski2023,
  title = {Machine learning coarse-grained potentials of protein thermodynamics},
  volume = {14},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-023-41343-1},
  DOI = {10.1038/s41467-023-41343-1},
  number = {1},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Majewski,  Maciej and Pérez,  Adrià and Th\"{o}lke,  Philipp and Doerr,  Stefan and Charron,  Nicholas E. and Giorgino,  Toni and Husic,  Brooke E. and Clementi,  Cecilia and Noé,  Frank and De Fabritiis,  Gianni},
  year = {2023},
  month = sep 
}
```

## Developer guide

### Implementing a new architecture

To implement a new architecture, you need to follow these steps:  
 **1.** Create a new class in `torchmdnet.models` that inherits from `torch.nn.Model`. Follow TorchMD_ET as a template. This is a minimum implementation of a model:  
```python 
class MyModule(nn.Module):
  def __init__(self, parameter1, parameter2):
	super(MyModule, self).__init__()
	# Define your model here
	self.layer1 = nn.Linear(10, 10)
	...
	# Initialize your model parameters here
	self.reset_parameters()

    def reset_parameters(self):
      # Initialize your model parameters here
	  nn.init.xavier_uniform_(self.layer1.weight)
	...
	
  def forward(self,
        z: Tensor, # Atomic numbers, shape (n_atoms, 1)
        pos: Tensor, # Atomic positions, shape (n_atoms, 3)
        batch: Tensor, # Batch vector, shape (n_atoms, 1). All atoms in the same molecule have the same value and are contiguous.
        q: Optional[Tensor] = None, # Atomic charges, shape (n_atoms, 1)
        s: Optional[Tensor] = None, # Atomic spins, shape (n_atoms, 1)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
	# Define your forward pass here
	scalar_features = ...
	vector_features = ...
	# Return the scalar and vector features, as well as the atomic numbers, positions and batch vector
	return scalar_features, vector_features, z, pos, batch
```
 **2.** Add the model to the `__all__` list in `torchmdnet.models.__init__.py`. This will make the tests pick your model up.  
 **3.** Tell models.model.create_model how to initialize your module by adding a new entry, for instance:  
 ```python
     elif args["model"] == "mymodule":
        from torchmdnet.models.torchmd_mymodule import MyModule
        is_equivariant = False # Set to True if your model is equivariant
        representation_model = MyModule(
            parameter1=args["parameter1"],
            parameter2=args["parameter2"],
            **shared_args, # Arguments typically shared by all models
        )
 ```
 
 **4.** Add any new parameters required to initialize your module to scripts.train.get_args. For instance:  
 ```python
   parser.add_argument('--parameter1', type=int, default=32, help='Parameter1 required by MyModule')
   ...
 ```
 **5.** Add an example configuration file to `torchmd-net/examples` that uses your model.  
 **6.** Make tests use your configuration file by adding a case to tests.utils.load_example_args. For instance:  
 ```python
 if model_name == "mymodule":
        config_file = join(dirname(dirname(__file__)), "examples", "MyModule-QM9.yaml")
 ```

At this point, if your module is missing some feature the tests will let you know, and you can add it. If you add a new feature to the package, please add a test for it.  

### Code style

We use [black](https://https://black.readthedocs.io/en/stable/). Please run `black` on your modified files before committing.  

### Testing

To run the tests, install the package and run `pytest` in the root directory of the repository. Tests are a good source of knowledge on how to use the different components of the package.  

