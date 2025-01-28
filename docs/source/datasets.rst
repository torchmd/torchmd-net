Datasets
========
.. toctree::
   :maxdepth: 3



Using a Dataset
---------------

There are two ways of using a dataset in TorchMD-Net, depending on whether you are using the Python API (for instance to run inferences) or the command line interface (for training).

Via a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~

You can make use of any of the available datasets via the :ref:`configuration file <configuration-file>` if you are using the :ref:`torchmd-train utility <torchmd-train>`.
Take a look at one of the example configuration files. As an example lets set up the :py:mod:`QM9` dataset, which allows for an additional argument `labels` specifying the subset to be provided. The part of the yaml configuration file for the dataset would look like this:

.. code:: yaml

   dataset: QM9
   dataset_arg:
     label: energy_U0



Via the Python API
~~~~~~~~~~~~~~~~~~

TorchMD-Net datasets are inherited from `torch Geometric datasets <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html>`_, you may use them whenever the PyTorch Geometric datasets are used. For instance, to load the QM9 dataset, you may use the following code:

.. code:: python

   from torchmdnet.datasets import QM9
   from torchmdnet.data import DataModule
   dataset = QM9(root='data', labels='energy_U0')
   
   print(dataset[0])
   print(len(dataset))
   # Some arbitrary parameters for the DataModule
   params = {'batch_size': 32,
	     'inference_batch_size': 32,
             'num_workers': 4,
	     'train_size': 0.8,
	     'val_size': 0.1,
	     'test_size': 0.1,
	     'seed': 42,
	     'log_dir': 'logs',
	     'splits': None,
	     'standardize': False,}
   
   dataloader = DataModule(params, dataset)
   dataloader.prepare_data()
   dataloader.setup("fit")

   # You can use this directly with PyTorch Lightning
   # trainer.fit(model, dataloader)

   
Adding a new Dataset
--------------------

If you want to train on custom data, first have a look at :py:mod:`torchmdnet.datasets.Custom`, which provides functionalities for loading a NumPy dataset consisting of atom types and coordinates, as well as energies, forces or both as the labels.

To add a new dataset, you need to:

1. Write a new class inheriting from :py:mod:`torch_geometric.data.Dataset`, see `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html>`_ for a tutorial on that. You may also start from :ref:`one of the base datasets we provide <base datasets>`.
2. Add the new class to the :py:mod:`torchmdnet.datasets` module (by listing it in the `__all__` variable in the `datasets/__init__.py` file) so that the :ref:`configuration file <configuration-file>` recognizes it.

.. note::

   The dataset must return torch-geometric `Data` objects, containing at least the keys `z` (atom types) and `pos` (atomic coordinates), as well as `y` (label), `neg_dy` (negative derivative of the label w.r.t atom coordinates) or both.


Datasets and Atomref
--------------------

Compatibility with the :py:mod:`torchmdnet.priors.Atomref` prior (and thus :ref:`delta learning <delta-learning>`) requires the dataset to define a method called :code:`get_atomref` as follows:

.. code:: python

   class MyDataset(Dataset):
       # ...
       def get_atomref(self, max_z=100):
          """Atomic energy reference values.
	  Args:
	      max_z (int): Maximum atomic number
	  Returns:
	      torch.Tensor: Atomic energy reference values for each element in the dataset.
	  """
          refs = pt.zeros(max_z)
	  # Set the references for each element present in the dataset.
	  refs[1] = -0.500607632585
	  refs[6] = -37.8302333826
	  refs[7] = -54.5680045287
	  refs[8] = -75.0362229210
	  # ...
          return refs.view(-1, 1)




Available Datasets
------------------

.. automodule:: torchmdnet.datasets
   :noindex:
      
   .. include:: generated/torchmdnet.datasets.rst
      :start-line: 5

.. _base datasets:
Base Datasets
-------------
The following datasets are used as a base for other datasets.

.. autoclass:: torchmdnet.datasets.ani.ANIBase
   :noindex:

.. autoclass:: torchmdnet.datasets.comp6.COMP6Base
   :noindex:

.. autoclass:: torchmdnet.datasets.memdataset.MemmappedDataset
   :noindex:
