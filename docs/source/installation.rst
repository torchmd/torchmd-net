Installation
============

TorchMD-Net is available as a pip package as well as in `conda-forge <https://conda-forge.org/>`_

As TorchMD-Net depends on PyTorch we need to add additional index URLs to the installation command as per `pytorch <https://pytorch.org/get-started/locally/>`_
 
.. code-block:: shell

   # The following will install TorchMD-Net with PyTorch CUDA 12.6 version
   ACCELERATOR=cu126 pip install torchmd-net --extra-index-url https://download.pytorch.org/whl/${ACCELERATOR}

where `ACCELERATOR` can be replaced with any of the available options in the above pytorch instructions
(i.e. `cu118`, `cu126`, `cu128`, `cu130`, `cpu`) depending on which version of pytorch should be installed.
Keep in mind tha the `cpu` versions are orders of magnitude slower than CUDA builds and should only be used 
for testing and not actual evaluation.

Alternatively it can be installed with conda or mamba with one of the following commands.
We recommend using `Miniforge <https://github.com/conda-forge/miniforge/>`_ instead of anaconda.

.. code-block:: shell

   # The following will install TorchMD-Net with PyTorch CUDA 12.6 version
   CUDA=12.6 mamba install torchmd-net cuda-version=${CUDA} -c conda-forge

Again here CUDA can be replaced with any version supported by the current pytorch version.
When installing with conda it will not automatically install Triton which is used for speeding out
neighbor calculations as it only exist on conda for Linux. If you are running on Linux and want the full
performance of torchmd-net also run `mamba install triton -c conda-forge`


Install from source
-------------------
For development purposes, we recommend using `uv <https://docs.astral.sh/uv/>`_ to install the TorchMD-Net in editable mode. 
After installing uv, run the following command to install the TorchMD-Net and its dependencies in editable mode.

.. code-block:: shell

   uv sync

This will install the package alongside PyTorch for CUDA 12.6. To change the CUDA version, 
edit the `pyproject.toml <https://github.com/torchmd/torchmd-net/blob/main/pyproject.toml>`_ file and change the `torch` dependency to the desired version.


