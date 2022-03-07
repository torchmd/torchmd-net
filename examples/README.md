# Examples

## Training
We provide three example config files for the ET for training on QM9, MD17 and ANI1 respectively. To train on a QM9 target other than `energy_U0`, change the parameter `dataset_arg` in the QM9 config file. Changing the MD17 molecule to train on works analogously. To train an ET from scratch you can use the following code from the torchmd-net directory:
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --conf examples/ET-{QM9,MD17,ANI1}.yaml
```
Use the `CUDA_VISIBLE_DEVICES` environment variable to select which and how many GPUs you want to train on. The example above selects GPUs with indices 0 and 1. The training code will want to save checkpoints and config files in a directory called `logs/`, which you can change either in the config .yaml file or as an additional command line argument: `--log-dir path/to/log-dir`.

## Loading checkpoints
You can access several pretrained checkpoint files under the following URLs:
- equivariant Transformer pretrained on QM9 (U0): http://pub.htmd.org/et-qm9.zip
- equivariant Transformer pretrained on MD17 (aspirin): http://pub.htmd.org/et-md17.zip
- equivariant Transformer pretrained on ANI1: http://pub.htmd.org/et-ani1.zip
- invariant Transformer pretrained on ANI1: http://pub.htmd.org/t-ani1.zip

The checkpoints can be loaded using the `load_model` function in TorchMD-Net. Additional model arguments (e.g. turning on force prediction on top of energies) for inference can also be passed to the function. See the following example code for loading an ET pretrained on the ANI1 dataset:
```python
from torchmdnet.models.model import load_model
model = load_model("ANI1-equivariant_transformer/epoch=359-val_loss=0.0004-test_loss=0.0120.ckpt ", derivative=True)
```
The following example shows how to run inference on the model checkpoint. For single molecules, you just have to pass atomic numbers and position tensors, to evaluate the model on multiple molecules simultaneously, also include a batch tensor, which contains the molecule index of each atom.
```python
import torch

# single molecule
z = torch.tensor([1, 1, 8], dtype=torch.long)
pos = torch.rand(3, 3)
energy, forces = model(z, pos)

# multiple molecules
z = torch.tensor([1, 1, 8, 1, 1, 8], dtype=torch.long)
pos = torch.rand(6, 3)
batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
energies, forces = model(z, pos, batch)
```
