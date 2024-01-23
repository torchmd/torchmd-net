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

2. Install the dependencies in environment.yml.

.. code-block:: shell

      conda env create -f environment.yml
      conda activate torchmd-net

3. CUDA enabled installation

You can skip this section if you only need a CPU installation.

You will need the CUDA compiler (nvcc) and the corresponding development libraries to build TorchMD-Net with CUDA support. You can install CUDA from the `official NVIDIA channel <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation>`_ or from conda-forge.

The conda-forge channel `changed the way to install CUDA from versions 12 and above <https://github.com/conda-forge/conda-forge.github.io/issues/1963>`_, thus the following instructions depend on whether you need CUDA < 12. If you have a GPU available, conda-forge probably installed the CUDA runtime (not the developer tools) on your system already, you can check with conda:
   
.. code-block:: shell

   conda list | grep cuda

   
Or by asking pytorch:
   
.. code-block:: shell
		 
   python -c "import torch; print(torch.version.cuda)"

   
It is recommended to install the same version as the one used by torch.  

.. warning:: At the time of writing there is a `bug in Mamba <https://github.com/mamba-org/mamba/issues/3120>`_ (v1.5.6) that can cause trouble when installing CUDA on an already created environment. We thus recommend conda for this step.
	     
* CUDA>=12

.. code-block:: shell

   conda install -c conda-forge cuda-nvcc cuda-libraries-dev cuda-version "gxx<12" pytorch=*=*cuda*

   
.. warning:: gxx<12 is required due to a `bug in GCC+CUDA12 <https://github.com/pybind/pybind11/issues/4606>`_ that prevents pybind11 from compiling correctly
	      

* CUDA<12  
  
The nvidia channel provides the developer tools for CUDA<12.
  
.. code-block:: shell
		 
    conda install -c nvidia "cuda-nvcc<12" "cuda-libraries-dev<12" "cuda-version<12" "gxx<12" pytorch=*=*cuda*


4. Install TorchMD-NET into the environment:

.. code-block:: shell

      pip install -e .


.. note:: Pip installation in CUDA mode requires compiling CUDA source codes, this can take a really long time and the process might appear as if it is "stuck". Run pip with `-vv` to see the compilation process.

This will install TorchMD-NET in editable mode, so that changes to the source code are immediately available.
Besides making all python utilities available environment-wide, this will also install the ``torchmd-train`` command line utility.

