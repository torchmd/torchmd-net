.. _neural-network-potentials:

Neural Network Potentials
=========================


.. figure:: img/tmdnet_model.*	    
   :align: center
      
   Schematic representation of the TorchMD-Net model.


In TorchMD-Net a model, abstracted by the :py:mod:`torchmdnet.models.model.TorchMD_Net` class, is composed of three main components:

1. **A representation model** that takes atomic numbers and positions (and optionally other per-sample or per-atom properties that the particular model might make use of) as input and outputs a series of per-atom features. The representation model is responsible for encoding the local environment of each atom.

2. **An output model** that takes the per-atom features and outputs a single per-batch label (i.e. total energy).
   
3. Optionally, :ref:`priors <Priors>` can be used to add additional constraints to the output model. For instance, a prior can be used to add a reference energy to the output of the model, or add a :ref:`Coulomb potential <:py:mod:torchmdnet.priors.Coulomb>` to the output of the model.

The resulting model can also be used to compute the negative gradient of the output with respect to the input positions (i.e. forces) via backpropagation with `autograd <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_. This is done by setting the :code:`derivative` flag to :code:`True` when creating the model.

.. hint:: Given the large amount of configuration options available, one typically does not instantiate :py:mod:`torchmdnet.models.model.TorchMD_Net` directly, but uses the :py:mod:`torchmdnet.models.model.create_model` function.

.. hint:: It is possible to use the :py:mod:`torchmdnet.models.model.TorchMD_Net` class directly instead of using the :py:mod:`torchmdnet.models.model.create_model` function. This can be useful if, for instance, you want to make use of the default parameters of the representation model and output model.
	  
	  .. code:: python

	     from torchmdnet.models.model import TorchMD_Net
	     from torchmdnet.models.tensornet import TensorNet
	     from torchmdnet.models.output_modules import Scalar

	     model = TorchMD_Net(
	     representation_model=TensorNet(),
	     output_model=Scalar(hidden_channels=32),
	     )


Training a model
----------------

The typical workflow to obtain a neural network potential in TorchMD-Net starts with :ref:`training <training>` one of the `Available Models`_. During this process you will get a checkpoint file that can be used to load the model for inference.

Custom training loops
~~~~~~~~~~~~~~~~~~~~~

If you want to use a custom training loop, you can use the :py:mod:`torchmdnet.models.model.TorchMD_Net` class directly. This class is a wrapper around the representation model, output model and priors that takes care of putting the pieces together.
In order to do this you need to follow these steps:

1. Create a new instance of :py:mod:`torchmdnet.models.model.TorchMD_Net` with the representation model, output model and priors you want to use. We provide the :py:mod:`torchmdnet.models.model.create_model` function to help you with this. Check :ref:`torchmd-train` for a list and description of the parameters.

2. Use one of the available :ref:`Datasets` to prepare your data, or create a custom one. You may then use a `pytorch DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders>`_ to iterate over the data.

3. Train the model using a custom training loop. See for instance the `PyTorch tutorial on training loops <https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#optimizer>`_.

This is a minimal example of a custom training loop:

.. code:: python

  import torch
  from torchmdnet.models.model import create_model
  from torchmdnet.datasets import MD17
  from torch_geometric.loader import DataLoader
  import yaml
  
  args = yaml.load(open("TensorNet-rMD17.yaml"), Loader=yaml.FullLoader)
  model = create_model(args, prior_model=None)
  dataset = MD17(root="~/data", molecules="revised_aspirin")
  if torch.cuda.is_available():
      model = model.to("cuda")
      dataset = dataset.to("cuda")
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  criterion = torch.nn.MSELoss()
  for epoch in range(10):
      for batch in dataloader:
          args = batch.to_dict()
          optimizer.zero_grad()
          y, neg_dy = model(args["z"], args["pos"], args["batch"])
          # A simple loss function that uses the energy and forces to train the model
          loss = criterion(y, args["y"]) + criterion(neg_dy, args["neg_dy"])
          loss.backward()
          optimizer.step()
  
   
	  
   
Loading a model for inference
-----------------------------

Once you have trained a model you should have a checkpoint that you can load for inference using :py:mod:`torchmdnet.models.model.load_model` as in the following example.

.. code:: python
	  
   import torch
   from torchmdnet.models.model import load_model
   checkpoint = "/path/to/checkpoint/my_checkpoint.ckpt"  
   model = load_model(checkpoint, derivative=True)
   # An arbitrary set of inputs for the model
   n_atoms = 10   
   zs = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)
   z = zs[torch.randint(0, len(zs), (n_atoms,))]
   pos = torch.randn(len(z), 3)
   batch = torch.zeros(len(z), dtype=torch.long)

   y, neg_dy = model(z, pos, batch)
   
.. note:: You can train a model using only the labels (i.e. energy) by passing :code:`derivative=False` and then set it to :code:`True` to compute its derivative (i.e. forces) only during inference.

.. note:: Some models take additional inputs such as the charge :code:`q` and the spin :code:`s` of the atoms depending on the chosen priors/outputs. Check the documentation of the model you are using to see if this is the case.

.. note:: When periodic boundary conditions are required, modules typically offer the possibility of providing the box vectors at construction and/or as an argument to the forward pass. Check the documentation of the class you are using to see if this is the case.


.. _delta-learning:
Training on relative energies
-----------------------------

It might be useful to train the model on relative energies but then make the model produce total energies when running inference.
TorchMD-Net supports delta training via the :code:`remove_ref_energy` option. Passing this option when training (either via the :ref:`configuration-file` or using the :ref:`torchmd-train` command line interface) will subtract the reference energy from each atom in a sample before passing it to the model.

.. note:: Delta learning requires a :ref:`dataset <Datasets>` that is compatible with :py:mod:`torchmdnet.priors.Atomref`.

If :code:`remove_ref_energy` is turned on, the reference energy is stored in the checkpoint file and is added back to the output of the model during inference if the model is loaded with :code:`remove_ref_energy=False`.

.. note:: The reference energies are stored as an :py:mod:`torchmdnet.priors.Atomref` prior with :code:`enable=False`.

Example
~~~~~~~

First we train a model with the :code:`remove_ref_energy` option turned on:

.. code:: shell

	  torchmd-train --config /path/to/config.yaml --remove_ref_energy

Then we load the model for inference:

.. code:: python

	  import torch
	  from torchmdnet.models.model import load_model
	  checkpoint = "/path/to/checkpoint/my_checkpoint.ckpt"  
	  model = load_model(checkpoint, remove_ref_energy=False)

	  # An arbitrary set of inputs for the model
	  n_atoms = 10   
	  zs = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)
	  z = zs[torch.randint(0, len(zs), (n_atoms,))]
	  pos = torch.randn(len(z), 3)
	  batch = torch.zeros(len(z), dtype=torch.long)

	  y, neg_dy = model(z, pos, batch)



Integration with MD packages
-----------------------------

It is possible to use the Neural Network Potentials in TorchMD-Net as force fields for Molecular Dynamics.

OpenMM
~~~~~~

The `OpenMM-Torch <https:\\github.com\openmm\openmm-torch>`_ plugin can be used to load :ref:`pretrained-models` as force fields in `OpenMM <https:\\github.com\openmm\openmm>`_. In order to do that one needs a translation layer between :py:mod:`TorchMD_Net <torchmdnet.models.model.TorchMD_Net>` and `TorchForce <https://github.com/openmm/openmm-torch?tab=readme-ov-file#exporting-a-pytorch-model-for-use-in-openmm>`_. This wrapper needs to take into account the different parameters and units (depending on the :ref:`Dataset <Datasets>` used to train the model) in both.

We provide here a minimal example of the wrapper class, but a complete example is provided under the `examples` folder.

.. code:: python

	import torch
	from torch import Tensor, nn
	import openmm
	import openmmtorch
	from torchmdnet.models.model import load_model
	# This is a wrapper that links an OpenMM Force with a TorchMD-Net model
	class Wrapper(nn.Module):
	
	    def __init__(self, embeddings: Tensor, checkpoint_path: str):
	        super(Wrapper, self).__init__()
		# The embeddings used to train the model, typically atomic numbers
	        self.embeddings = embeddings
		# We let OpenMM compute the forces from the energies
	        self.model = load_model(checkpoint_path, derivative=False)
	
	    def forward(self, positions: Tensor) -> Tensor:
	        # OpenMM works with nanometer positions and kilojoule per mole energies
	        # Depending on the model, you might need to convert the units
	        positions = positions.to(torch.float32) * 10.0 # nm -> A
	        energy = self.model(z=self.embeddings, pos=positions)[0]
	        return energy * 96.4916 # eV -> kJ/mol

	model = Wrapper(embeddings=torch.tensor([1, 6, 7, 8, 9]), checkpoint_path="/path/to/checkpoint/my_checkpoint.ckpt")
	model = torch.jit.script(model) # Models need to be scripted to be used in OpenMM
	# The model can be used as a force field in OpenMM
	force = openmmtorch.TorchForce(model)
	# Add the force to an OpenMM system
	system = openmm.System()
	system.addForce(force)

	
.. note:: See :ref:`training <training>` for more information on how to train a model.

.. warning:: The conversion factors are specific to the dataset used to train the model. Check the documentation of the dataset you are using to see if this is the case.

.. note:: See the `OpenMM-Torch <https:\\github.com\openmm\openmm-torch>`_ documentation for more information on additional functionality (such as periodic boundary conditions or CUDA graph support).


TorchMD
~~~~~~~

Integration with `TorchMD <https:\\github.com\torchmd\torchmd>`_ is carried out via :py:mod:`torchmdnet.calculators.External`. Refer to its documentation for more information on additional functionality.

.. code:: python

	import torch
	import torchmd
	from torchmdnet.calculators import External
	# Load the model
	embeddings = torch.tensor([1, 6, 7, 8, 9])
	model = External("/path/to/checkpoint/my_checkpoint.ckpt, embeddings)
	# Use the calculator in a TorchMD simulation
	from torchmd.forces import Forces
	parameters = # Your TorchMD parameters here
	torchmd_forces = Forces(parameters, external=model)
	# You can now pass torchmd_forces to a TorchMD Integrator

Additionally, the calculator can be specified in the configuration file of a TorchMD simulation via the `external` key.


.. code:: yaml
	  
	...
	external:
	  module: torchmdnet.calculators
	  file: /path/to/checkpoint/my_checkpoint.ckpt
	  embeddings: [1, 6, 7, 8, 9]
	...

.. warning:: Unit conversion might be required depending on the dataset used to train the model. Check the documentation of the dataset you are using to see if this is the case.

Available Models
----------------

TorchMD-Net offers representation models that output a series of per-atom features. Typically one wants to couple this with an :py:mod:`output model <torchmdnet.models.output_modules>` and perhaps a :py:mod:`prior <torchmdnet.priors>` to get a single per-batch label (i.e. total energy) and optionally its derivative with respect to the positions (i.e. forces).

The :py:mod:`TorchMD_Net <torchmdnet.models.model.TorchMD_Net>` model takes care of putting the pieces together.


TensorNet
~~~~~~~~~

TensorNet is an equivariant model based on rank-2 Cartesian tensor representations. Euclidean neural network potentials have been shown to achieve state-of-the-art performance and better data efficiency than previous models, relying on higher-rank equivariant features which are irreducible representations of the rotation group, in the form of spherical tensors. However, the computation of tensor products in these models can be computationally demanding. In contrast, TensorNet exploits the use of Cartesian rank-2 tensors (3x3 matrices) which can be very efficiently decomposed into scalar, vector and rank-2 tensor features. Furthermore, Clebsch-Gordan tensor products are substituted by straightforward 3x3 matrix products. Overall, these properties allow TensorNet to achieve state-of-the-art accuracy on common benchmark datasets such as rMD17 and QM9 with a reduced number of message passing steps, learnable parameters and computational cost. The prediction of up to rank-2 molecular properties that behave appropriately under geometric transformations such as reflections and rotations is also possible.

.. automodule:: torchmdnet.models.tensornet
   :noindex:

.. note:: TensorNet is referred to as "tensornet" in the :ref:`configuration-file`.
	  
Equivariant Transformer
~~~~~~~~~~~~~~~~~~~~~~~

The Equivariant Transformer (ET) is an equivariant neural network which uses both scalar and Cartesian vector representations. The distinctive feature of the ET in comparison to other Cartesian vector models such as PaiNN or EGNN is the use of a distance-dependent dot product attention mechanism, which achieved state-of-the-art performance on benchmark datasets at the time of publication. Furthermore, the analysis of attention weights allowed to extract insights on the interaction of different atomic species for the prediction of molecular energies and forces. The model also exhibits a low computational cost for inference and training in comparison to some of the most used NNPs in the literature.

.. automodule:: torchmdnet.models.torchmd_et
   :noindex:

      
.. note:: Equivariant Transformer is referred to as "equivariant-transformer" in the :ref:`configuration-file`.
	  
Graph Network
~~~~~~~~~~~~~

The graph network is an invariant model inspired on the SchNet and PhysNet architectures. The network was optimized to have satisfactory performance on coarse-grained proteins, allowing to build NNPs that correctly reproduce protein free energy landscapes. In contrast to the ET and TensorNet, the graph network only uses relative distances between atoms as geometrical information, which are invariant to translations, rotations and reflections. The distances are used by the model to learn a set of continuous filters that are applied to feature graph convolutions as in SchNet, progressively updating the intial atomic embeddings by means of residual connections.

.. automodule:: torchmdnet.models.torchmd_gn
   :noindex:


.. note:: Graph Network is referred to as "graph-network" in the :ref:`configuration-file`.

Implementing a new Architecture
-------------------------------

To implement a new architecture, you need to follow these steps:

1. Create a new class in ``torchmdnet.models`` that inherits from ``torch.nn.Model``. Follow TorchMD_ET as a template. This is a minimum implementation of a model:

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
                
            def forward(self, z: Tensor, pos: Tensor, batch: Tensor, box: Optional[Tensor], q: Optional[Tensor] = None, s: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                # Define your forward pass here
                scalar_features = ...
                vector_features = ...
                return scalar_features, vector_features, z, pos, batch

2. Add the model to the ``__all__`` list in ``torchmdnet.models.__init__.py``. This will make the tests pick your model up.

3. Tell models.model.create_model how to initialize your module by adding a new entry:

    .. code-block:: python

        elif args["model"] == "mymodule":
            from torchmdnet.models.torchmd_mymodule import MyModule
            is_equivariant = False
            representation_model = MyModule(
                parameter1=args["parameter1"],
                parameter2=args["parameter2"],
                **shared_args,
            )

4. Add any new parameters required to initialize your module to scripts.train.get_args:

    .. code-block:: python

        parser.add_argument('--parameter1', type=int, default=32, help='Parameter1 required by MyModule')

5. Add an example configuration file to ``torchmd-net/examples`` that uses your model.

6. Make tests use your configuration file by adding a case to tests.utils.load_example_args:

    .. code-block:: python

        if model_name == "mymodule":
            config_file = join(dirname(dirname(__file__)), "examples", "MyModule-QM9.yaml")

At this point, if your module is missing some feature the tests will let you know, and you can add it. If you add a new feature to the package, please add a test for it.
	 
