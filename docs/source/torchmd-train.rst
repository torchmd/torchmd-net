.. _torchmd-train:

TorchMD-Train Utility
---------------------

.. _configuration-file:

Configuration file
~~~~~~~~~~~~~~~~~~

The torchmd-train utility can be configured via a `yaml <https://en.wikipedia.org/wiki/YAML>`_ file, see below for a list of available options. You can include any valid option in the yaml file by replacing "-" by "_", for instance:

.. code:: yaml

   activation: silu
   aggr: add
   atom_filter: -1
   batch_size: 16
   coord_files: null
   cutoff_lower: 0.0
   cutoff_upper: 5.0
   dataset: QM9
   dataset_arg:
     label: energy_U0
   dataset_root: ~/data
   derivative: false
   early_stopping_patience: 150
   ema_alpha_neg_dy: 1.0
   ema_alpha_y: 1.0
   embed_files: null
   embedding_dimension: 256
   energy_files: null
   equivariance_invariance_group: O(3)
   y_weight: 1.0
   force_files: null
   neg_dy_weight: 0.0
   gradient_clipping: 40
   inference_batch_size: 128
   load_model: null
   log_dir: logs/
   lr: 0.0001
   lr_factor: 0.8
   lr_min: 1.0e-07
   lr_patience: 15
   lr_warmup_steps: 1000
   max_num_neighbors: 64
   max_z: 128
   model: tensornet
   ngpus: -1
   num_epochs: 3000
   num_layers: 3
   num_nodes: 1
   num_rbf: 64
   num_workers: 6
   output_model: Scalar
   precision: 32
   prior_model: Atomref
   rbf_type: expnorm
   redirect: false
   reduce_op: add
   save_interval: 10
   seed: 1
   splits: null
   standardize: false
   test_interval: 20
   test_size: null
   train_size: 110000
   trainable_rbf: false
   val_size: 10000
   weight_decay: 0.0
   charge: false
   spin: false
   
.. note:: There are several example files in the `examples/` folder.

You can use a yaml configuration file with the `torchmd-train` utility with:

.. code:: bash

   torchmd-train --conf my_conf.yaml 

.. note:: Flags provided after `--conf` will override the ones in the yaml file.

.. note::

   The utility will save all the provided parameters as a file called `input.yaml` along with the generated checkpoints.

Command line interface
~~~~~~~~~~~~~~~~~~~~~~


.. autoprogram:: scripts.train:get_argparse()
   :prog: torchmd-train

	     
