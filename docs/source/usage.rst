Usage
-----

.. _training:

Training an existing model
==========================

Specifying training arguments can either be done via a :ref:`configuration YAML file <configuration-file>` or through command line arguments directly, see the :ref:`torchmd-train <torchmd-train>` utility for more info. Several examples of architectural and training specifications for some models and datasets can be found in `examples <https://github.com/torchmd/torchmd-net/tree/main/examples>`_.

GPUs can be selected by setting the `CUDA_VISIBLE_DEVICES` environment variable. Otherwise, the argument `--ngpus` can be used to select the number of GPUs to train on (-1, the default, uses all available GPUs or the ones specified in `CUDA_VISIBLE_DEVICES`).


.. note::

   Keep in mind that the `GPU ID reported by nvidia-smi might not be the same as the one CUDA_VISIBLE_DEVICES uses <https://stackoverflow.com/questions/26123252/inconsistency-of-ids-between-nvidia-smi-l-and-cudevicegetname>`_.

For example, to train the Equivariant Transformer on the QM9 dataset with the architectural and training hyperparameters described in the paper, one can run

.. code:: bash

    mkdir output
    CUDA_VISIBLE_DEVICES=0 torchmd-train --conf torchmd-net/examples/ET-QM9.yaml --log-dir output/

Run `torchmd-train --help` to see all available options and their descriptions.

Pretrained Models
=================

See `here <https://github.com/torchmd/torchmd-net/tree/main/examples#loading-checkpoints>`_ for instructions on how to load pretrained models.

Custom Prior Models
===================

In addition to implementing a custom dataset class, it is also possible to add a custom prior model to the model. This can be done by implementing a new prior model class in :py:mod:`torchmdnet.priors` and adding the argument ``--prior-model <PriorModelName>``. As an example, have a look at :py:mod:`torchmdnet.priors.Atomref`.

Periodic Boundary Conditions
============================

TorchMD-Net supports periodic boundary conditions with arbitrary triclinic boxes.

Periodic boundary conditions can be enabled in several ways, depending on how you are using TorchMD-Net:
#. Pass the `box-vecs` option in the :ref:`configuration file <configuration-file>`.
#. Pass the ``--box-vecs`` argument to the :ref:`torchmd-train <torchmd-train>` utility.
#. Choose or write a dataset that provides a box for each sample. See for instance the :py:mod:`torchmdnet.datasets.WaterBox` dataset.
#. You may also send the box vectors directly to a :ref:`neural network potential <neural-network-potentials>` as an argument when running inference, e.g. ``model(z, pos, batch, box=box_vecs)``.


For a given cutoff, :math:`r_c`, the box vectors :math:`\vec{a},\vec{b},\vec{c}` must satisfy certain requirements:

.. math::
	  
  \begin{align*}
  a_y = a_z = b_z &= 0 \\
  a_x, b_y, c_z &\geq 2 r_c \\
  a_x &\geq 2  b_x \\
  a_x &\geq 2  c_x \\
  b_y &\geq 2  c_y
  \end{align*}

These requirements correspond to a particular rotation of the system and reduced form of the vectors, as well as the requirement that the cutoff be no larger than half the box width.

.. note:: The box defined by the vectors :math:`\vec{a} = (L_x, 0, 0)`, :math:`\vec{b} = (0, L_y, 0)`, and :math:`\vec{c} = (0, 0, L_z)` correspond to a rectangular box. In this case, the input option in the :ref:`configuration file <configuration-file>` would be ``box-vecs: [[L_x, 0, 0], [0, L_y, 0], [0, 0, L_z]]``.


CUDA Graphs
============

TensorNet is capturable into a `CUDA graph <https://developer.nvidia.com/blog/cuda-graphs/>`_ with the right options. This can dramatically increase performance during inference. The dynamically-shaped nature of training makes CUDA graphs not an option in most practical cases.

For TensorNet to be CUDA-graph compatible, `check_errors` must be `False` and `static_shapes` must be `True`. Manually capturing a piece of code can be challenging, instead, to take advantage of CUDA graphs you can use :py:mod:`torchmdnet.calculators.External`, which helps integrating a Torchmd-NET model into another code, or `OpenMM-Torch <https://github.com/openmm/openmm-torch>`_ if you are using OpenMM.



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

	  
Developer Guide
---------------

Code Style
==========

We use `black <https://black.readthedocs.io/en/stable/>`_. Please run ``black`` on your modified
