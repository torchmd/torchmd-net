# AceFF


Pretained AceFF models are available for use with TorchMD-Net.

The models can be downloaded from HuggingFace https://huggingface.co/collections/Acellera/aceff-machine-learning-potentials

They can be used stand-alone with torchmd-net:
```python
import torch
from torchmdnet.models.model import load_model
from huggingface_hub import hf_hub_download

model_file_path = hf_hub_download(
    repo_id="Acellera/AceFF-2.0",
    filename="aceff_v2.0.ckpt"
)

model = load_model(model_file_path, derivative=True)

z = torch.tensor([1, 1, 8], dtype=torch.long)
pos = torch.rand(3, 3)
energy, forces = model(z, pos)

```

## ASE calculator
They can also be used with the torchmd-net ASE calculator

```python
from ase.build import molecule
from torchmdnet.calculators import TMDNETCalculator
from huggingface_hub import hf_hub_download

model_file_path = hf_hub_download(
    repo_id="Acellera/AceFF-2.0",
    filename="aceff_v2.0.ckpt"
)

calc  = TMDNETCalculator(model_file_path, device='cuda')

atoms = molecule('C2H6')
atoms.calc = calc

# The total molecular charge must be set 
atoms.info['charge'] = 0

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```


See the example python scripts in this folder for more information.


