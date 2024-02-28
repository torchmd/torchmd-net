Welcome to the TorchMD-NET Documentation!
=========================================

TorchMD-NET provides state-of-the-art neural networks potentials (NNPs) and a mechanism to train them. It offers efficient and fast implementations of several NNPs and is integrated with GPU-accelerated molecular dynamics code like `ACEMD <https://www.acellera.com/products/molecular-dynamics-software-gpu-acemd/>`_, `OpenMM <https://www.openmm.org>`_, and `TorchMD <https://github.com/torchmd/torchmd>`_. TorchMD-NET exposes its NNPs as `PyTorch <https://pytorch.org>`_ modules.


Cite
====

If you use TorchMD-NET in your research, please cite the following papers:

Main reference
~~~~~~~~~

.. code-block:: bibtex

    @misc{
    pelaez2024torchmdnet,
    title={TorchMD-Net 2.0: Fast Neural Network Potentials for Molecular Simulations},
    author={Raul P. Pelaez and Guillem Simeon and Raimondas Galvelis and Antonio Mirarchi and Peter Eastman and Stefan Doerr and Philipp Thölke and Thomas E. Markland and Gianni De Fabritiis},
    year={2024},
    eprint={2402.17660},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
    }

TensorNet
~~~~~~~~~

.. code-block:: bibtex

    @inproceedings{
    simeon2023tensornet,
    title={TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials},
    author={Guillem Simeon and Gianni De Fabritiis},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=BEHlPdBZ2e}
    }


Equivariant Transformer
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

  @article{Majewski2023,
  title = {Machine learning coarse-grained potentials of protein thermodynamics},
  volume = {14},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-023-41343-1},
  DOI = {10.1038/s41467-023-41343-1},
  number = {1},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Majewski,  Maciej and Pérez,  Adrià and Th\"{o}lke,  Philipp and Doerr,  Stefan and Charron,  Nicholas E. and Giorgino,  Toni and Husic,  Brooke E. and   Clementi,  Cecilia and Noé,  Frank and De Fabritiis,  Gianni},
  year = {2023},
  month = sep 
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
