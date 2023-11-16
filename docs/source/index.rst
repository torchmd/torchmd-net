Welcome to the TorchMD-NET Documentation!
=========================================

TorchMD-NET provides state-of-the-art neural networks potentials (NNPs) and a mechanism to train them. It offers efficient and fast implementations of several NNPs and is integrated with GPU-accelerated molecular dynamics code like `ACEMD <https://www.acellera.com/products/molecular-dynamics-software-gpu-acemd/>`_, `OpenMM <https://www.openmm.org>`_, and `TorchMD <https://github.com/torchmd/torchmd>`_. TorchMD-NET exposes its NNPs as `PyTorch <https://pytorch.org>`_ modules.


Cite
====

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


.. toctree::
   :hidden:
      
   installation
   usage
   torchmd-train
   datasets
   models
   priors
   api

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
