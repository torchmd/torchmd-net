Installation
============

TorchMD-Net is available in `conda-forge <https://conda-forge.org/>`_ and can be installed with:

.. code-block:: shell

   mamba install torchmd-net

We recommend using `Mamba <https://github.com/conda-forge/miniforge/#mambaforge>`_ instead of conda.

Install from source
-------------------

1. Clone the repository:

   .. code-block:: shell

      git clone https://github.com/torchmd/torchmd-net.git
      cd torchmd-net

2. Install the dependencies in environment.yml. You can do it via pip, but we recommend `Mambaforge <https://github.com/conda-forge/miniforge/#mambaforge>`_ instead.

3. Create an environment and activate it:

   .. code-block:: shell

      mamba env create -f environment.yml
      mamba activate torchmd-net

4. Install TorchMD-NET into the environment:

   .. code-block:: shell

      pip install -e .

This will install TorchMD-NET in editable mode, so that changes to the source code are immediately available.
Besides making all python utilities available environment-wide, this will also install the ``torchmd-train`` command line utility.

CUDA enabled installation
-------------------------

Besides the dependencies listed in the environment file, you will also need the CUDA ``nvcc`` compiler suite to build TorchMD-Net.
If your system lacks nvcc you may install it via conda-forge:

.. code-block:: shell

   mamba install cudatoolkit-dev

Or from the nvidia channel:

.. code-block:: shell

   mamba install -c nvidia cuda-nvcc cuda-cudart-dev cuda-libraries-dev

Make sure you install a major version compatible with your torch installation, which you can check with:

.. code-block:: shell

   python -c "import torch; print(torch.version.cuda)"
