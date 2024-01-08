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

4. CUDA enabled installation

   You can skip this section if you only need a CPU installation.

 .. code-block:: shell

   mamba install cuda-nvcc cuda-libraries-dev cuda-version "gxx<12" pytorch=*=*cuda*

 .. warning:: gxx<12 is required due to a `bug in GCC+CUDA12 <https://github.com/pybind/pybind11/issues/4606>`_ that prevents pybind11 from compiling correctly
	      
5. Install TorchMD-NET into the environment:

   .. code-block:: shell

      pip install -e .

This will install TorchMD-NET in editable mode, so that changes to the source code are immediately available.
Besides making all python utilities available environment-wide, this will also install the ``torchmd-train`` command line utility.

