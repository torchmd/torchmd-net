Welcome to the TorchMD-NET Documentation!
=========================================

TorchMD-NET provides state-of-the-art neural networks potentials (NNPs) and a mechanism to train them. It offers efficient and fast implementations of several NNPs and is integrated with GPU-accelerated molecular dynamics code like `ACEMD <https://www.acellera.com/products/molecular-dynamics-software-gpu-acemd/>`_, `OpenMM <https://www.openmm.org>`_, and `TorchMD <https://github.com/torchmd/torchmd>`_. TorchMD-NET exposes its NNPs as `PyTorch <https://pytorch.org>`_ modules.

Available Architectures
-----------------------

- `Equivariant Transformer (ET) <https://arxiv.org/abs/2202.02541>`_
- `Transformer (T) <https://arxiv.org/abs/2202.02541>`_
- `Graph Neural Network (GN) <https://arxiv.org/abs/2212.07492>`_
- `TensorNet <https://arxiv.org/abs/2306.06482>`_

Installation
------------

1. Clone the repository::

    git clone https://github.com/torchmd/torchmd-net.git
    cd torchmd-net

2. Install `Mambaforge <https://github.com/conda-forge/miniforge/#mambaforge>`_. We recommend using `mamba` rather than `conda`.

3. Create an environment and activate it::

    mamba env create -f environment.yml
    mamba activate torchmd-net

4. Install TorchMD-NET into the environment::

    pip install -e .

This will install TorchMD-NET in editable mode, so that changes to the source code are immediately available. Besides making all Python utilities available environment-wide, this will also install the `torchmd-train` command line utility.


Cite
----

If you use TorchMD-NET in your research, please cite the following papers:

Main reference
~~~~~~~~~~~~~~

.. code-block:: bibtex

    @inproceedings{
        tholke2021equivariant,
        title={Equivariant Transformers for Neural Network based Molecular Potentials},
        author={Philipp Th{\"o}lke and Gianni De Fabritiis},
        booktitle={International Conference on Learning Representations},
        year={2022},
        url={https://openreview.net/forum?id=zNHzqZ9wrRB}
    }

Graph Network
~~~~~~~~~~~~~

.. code-block:: bibtex

    @misc{majewski2022machine,
          title={Machine Learning Coarse-Grained Potentials of Protein Thermodynamics},
          author={Maciej Majewski and Adrià Pérez and Philipp Thölke and Stefan Doerr and Nicholas E. Charron and Toni Giorgino and Brooke E. Husic and Cecilia Clementi and Frank Noé and Gianni De Fabritiis},
          year={2022},
          eprint={2212.07492},
          archivePrefix={arXiv},
          primaryClass={q-bio.BM}
    }

TensorNet
~~~~~~~~~

.. code-block:: bibtex

    @misc{simeon2023tensornet,
          title={TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials},
          author={Guillem Simeon and Gianni de Fabritiis},
          year={2023},
          eprint={2306.06482},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

Developer Guide
---------------

Implementing a New Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement a new architecture, you need to follow these steps:

1. Create a new class in ``torchmdnet.models`` that inherits from ``torch.nn.Model``. Follow TorchMD_ET as a template. This is a minimum implementation of a model::

    .. code-block:: python

        class MyModule(nn.Module):
            def __init__(self, parameter1, parameter2):
                super(MyModule, self).__init__()
                # Define your model here
                self.layer1 = nn.Linear(10, 10)
                # Initialize your model parameters here
                self.reset_parameters()

            def reset_parameters(self):
                # Initialize your model parameters here
                nn.init.xavier_uniform_(self.layer1.weight)
                
            def forward(self, z: Tensor, pos: Tensor, batch: Tensor, q: Optional[Tensor] = None, s: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                # Define your forward pass here
                scalar_features = ...
                vector_features = ...
                return scalar_features, vector_features, z, pos, batch

2. Add the model to the ``__all__`` list in ``torchmdnet.models.__init__.py``. This will make the tests pick your model up.

3. Tell models.model.create_model how to initialize your module by adding a new entry::

    .. code-block:: python

        elif args["model"] == "mymodule":
            from torchmdnet.models.torchmd_mymodule import MyModule
            is_equivariant = False
            representation_model = MyModule(
                parameter1=args["parameter1"],
                parameter2=args["parameter2"],
                **shared_args,
            )

4. Add any new parameters required to initialize your module to scripts.train.get_args::

    .. code-block:: python

        parser.add_argument('--parameter1', type=int, default=32, help='Parameter1 required by MyModule')

5. Add an example configuration file to ``torchmd-net/examples`` that uses your model.

6. Make tests use your configuration file by adding a case to tests.utils.load_example_args::

    .. code-block:: python

        if model_name == "mymodule":
            config_file = join(dirname(dirname(__file__)), "examples", "MyModule-QM9.yaml")

At this point, if your module is missing some feature the tests will let you know, and you can add it. If you add a new feature to the package, please add a test for it.

Code Style
~~~~~~~~~~

We use `black <https://black.readthedocs.io/en/stable/>`_. Please run ``black`` on your modified

Contents
--------

.. toctree::

   usage
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
