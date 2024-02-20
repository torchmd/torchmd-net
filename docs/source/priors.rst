Priors
======

.. toctree::
   :maxdepth: 4


Priors in the context of TorchMD-Net are pre-defined models that embed domain-specific knowledge into the neural network. They are used to enforce known physical laws or empirical observations that are not automatically learned by the neural network. This inclusion enhances the predictive accuracy of the network, particularly in scenarios where training data may be limited or where the network needs to generalize beyond the scope of its training set.

The primary role of priors is to guide the learning process of the network by imposing constraints based on physical principles. For example, a prior might reflect known chemical properties or sum a Coulomb interaction to the energy predicted by the network.


Using Priors in TorchMD-Net
---------------------------

There are two ways of using a dataset in TorchMD-Net, depending on whether you are using the Python API (for instance to run inferences) or the command line interface (for training).

Via a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~

You can make use of any of the available priors via the :ref:`configuration file <configuration-file>` if you are using the :ref:`torchmd-train utility <torchmd-train>`.
In the YAML configuration file, you can specify the type of prior model to use along with any additional arguments it requires (see the documentation for each particular prior).

For example, to use the Atomref prior via YAML, your configuration might look like this:

.. code:: yaml

	  prior_model: Atomref

It is possible to configure more than one prior in this way:
	     
.. code:: yaml

    prior_model:
        Atomref: {} # No additional arguments
        Coulomb:
            lower_switch_distance: 4
            upper_switch_distance: 8
            max_num_neighbors: 128


	  
Via the Python API
~~~~~~~~~~~~~~~~~~

If you are using the Python API, you can use any of the available priors by importing them from the :mod:`torchmdnet.priors` module and passing them to the :class:`torchmdnet.models.TorchMDNet` class.

Writing a new Prior
--------------------

All Priors inherit from the :py:mod:`BasePrior` class. 

As an example, lets write a prior that adds an offset to the energy of each atom and molecule. We will call it :py:class:`EnergyOffset`.

.. code:: python
	  
   from torchmdnet.priors.base import BasePrior
   class EnergyOffset(BasePrior):

    def __init__(self, atom_offset=0, molecule_offset=0, dataset=None):
        super().__init__()
	self.atom_offset = atom_offset
	self.molecule_offset = molecule_offset

    def get_init_args(self):
        r"""A function that returns all required arguments to construct a prior object.
        The values should be returned inside a dict with the keys being the arguments' names.
        All values should also be saveable in a .yaml file as this is used to reconstruct the
        prior model from a checkpoint file.
        """
	return {"atom_offset": self.atom_offset, "molecule_offset": self.molecule_offset}
	
    def pre_reduce(self, x: Tensor, z: Tensor, pos: Tensor, batch: Tensor, extra_args: Optional[Dict[str, Tensor]]):
        """Adds the offset to the energy of each atom.
        """
        return x + self.atom_offset
	
    def post_reduce(self, x: Tensor, z: Tensor, pos: Tensor, batch: Tensor, extra_args: Optional[Dict[str, Tensor]]):
        """Adds the offset to the energy of each molecule.
        """
        return x + self.molecule_offset


Available Priors
----------------

.. automodule:: torchmdnet.priors
   :noindex:
      
   .. include:: generated/torchmdnet.priors.rst
      :start-line: 5

