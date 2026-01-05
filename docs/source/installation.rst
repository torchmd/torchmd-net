Installation
============

TorchMD-Net is available as a pip package as well as in `conda-forge <https://conda-forge.org/>`_

As TorchMD-Net depends on PyTorch we need to add additional index URLs to the installation command as per `pytorch <https://pytorch.org/get-started/locally/>`_

.. code-block:: shell

   # The following will install TorchMD-Net with PyTorch CUDA 11.8 version
   pip install torchmd-net --extra-index-url https://download.pytorch.org/whl/cu118
   # The following will install TorchMD-Net with PyTorch CUDA 12.4 version
   pip install torchmd-net --extra-index-url https://download.pytorch.org/whl/cu124
   # The following will install TorchMD-Net with PyTorch CPU only version (not recommended)
   pip install torchmd-net --extra-index-url https://download.pytorch.org/whl/cpu

Alternatively it can be installed with conda or mamba with one of the following commands.
We recommend using `Miniforge <https://github.com/conda-forge/miniforge/>`_ instead of anaconda.

.. code-block:: shell

   mamba install torchmd-net cuda-version=11.8
   mamba install torchmd-net cuda-version=12.4


Install from source
-------------------
For development purposes, we recommend using `uv <https://docs.astral.sh/uv/>`_ to install the TorchMD-Net in editable mode. 
After installing uv, run the following command to install the TorchMD-Net and its dependencies in editable mode.

.. code-block:: shell

   uv sync

This will install the package alongside PyTorch for CUDA 12.6. To change the CUDA version, 
edit the `pyproject.toml <https://github.com/torchmd/torchmd-net/blob/main/pyproject.toml>`_ file and change the `torch` dependency to the desired version.


