Usage
-----

Training an existing model
==========================

Specifying training arguments can either be done via a configuration YAML file or through command line arguments directly. Several examples of architectural and training specifications for some models and datasets can be found in `examples <https://github.com/torchmd/torchmd-net/tree/main/examples>`_.

.. note::

   If a parameter is present both in the YAML file and the command line, the command line version takes precedence. 



GPUs can be selected by setting the `CUDA_VISIBLE_DEVICES` environment variable. Otherwise, the argument `--ngpus` can be used to select the number of GPUs to train on (-1, the default, uses all available GPUs or the ones specified in `CUDA_VISIBLE_DEVICES`).


.. note::

   Keep in mind that the `GPU ID reported by nvidia-smi might not be the same as the one CUDA_VISIBLE_DEVICES uses <https://stackoverflow.com/questions/26123252/inconsistency-of-ids-between-nvidia-smi-l-and-cudevicegetname>`_.

For example, to train the Equivariant Transformer on the QM9 dataset with the architectural and training hyperparameters described in the paper, one can run::

    mkdir output
    CUDA_VISIBLE_DEVICES=0 torchmd-train --conf torchmd-net/examples/ET-QM9.yaml --log-dir output/

Run `torchmd-train --help` to see all available options and their descriptions.

Pretrained Models
=================

See `here <https://github.com/torchmd/torchmd-net/tree/main/examples#loading-checkpoints>`_ for instructions on how to load pretrained models.

Creating a New Dataset
======================

If you want to train on custom data, first have a look at ``torchmdnet.datasets.Custom``, which provides functionalities for loading a NumPy dataset consisting of atom types and coordinates, as well as energies, forces or both as the labels. Alternatively, you can implement a custom class according to the torch-geometric way of implementing a dataset. That is, derive the `Dataset` or `InMemoryDataset` class and implement the necessary functions (more info `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-your-own-datasets>`_). The dataset must return torch-geometric `Data` objects, containing at least the keys `z` (atom types) and `pos` (atomic coordinates), as well as `y` (label), `neg_dy` (negative derivative of the label w.r.t atom coordinates) or both.

Custom Prior Models
===================

In addition to implementing a custom dataset class, it is also possible to add a custom prior model to the model. This can be done by implementing a new prior model class in ``torchmdnet.priors`` and adding the argument ``--prior-model <PriorModelName>``. As an example, have a look at ``torchmdnet.priors.Atomref``.

This is a reference to the API page for the priors: :py:mod:`torchmdnet.priors`.

Multi-Node Training
===================

In order to train models on multiple nodes some environment variables have to be set, which provide all necessary information to PyTorch Lightning. In the following, we provide an example bash script to start training on two machines with two GPUs each. The script has to be started once on each node. Once ``torchmd-train`` is started on all nodes, a network connection between the nodes will be established using NCCL.

.. code-block:: shell

    export NODE_RANK=0
    export MASTER_ADDR=hostname1
    export MASTER_PORT=12910

    mkdir -p output
    CUDA_VISIBLE_DEVICES=0,1 torchmd-train --conf torchmd-net/examples/ET-QM9.yaml.yaml --num-nodes 2 --log-dir output/

- ``NODE_RANK`` : Integer indicating the node index. Must be `0` for the main node and incremented by one for each additional node.
- ``MASTER_ADDR`` : Hostname or IP address of the main node. The same for all involved nodes.
- ``MASTER_PORT`` : A free network port for communication between nodes. PyTorch Lightning suggests port `12910` as a default.

.. admonition:: Known Limitations
	  
	  - Due to the way PyTorch Lightning calculates the number of required DDP processes, all nodes must use the same number of GPUs. Otherwise training will not start or crash.
	  - We observe a 50x decrease in performance when mixing nodes with different GPU architectures (tested with RTX 2080 Ti and RTX 3090).
