# This script shows how to load the AceFF-1.0 checkpoint
# and use it for force and energy predictions for a single molecule or a batch
import torch


# The AceFF-1.0 model is available from HuggingFace under apache2.0 license
from huggingface_hub import hf_hub_download

model_file_path = hf_hub_download(
    repo_id="Acellera/AceFF-2.0", filename="aceff_v2.0.ckpt"
)

print("Downloaded to:", model_file_path)

from torchmdnet.models.model import load_model

model = load_model(model_file_path, derivative=True)


# single molecule
z = torch.tensor([1, 1, 8], dtype=torch.long)
pos = torch.rand(3, 3)
energy, forces = model(z, pos)
print(energy)
print(forces)

# multiple molecules
z = torch.tensor([1, 1, 8, 1, 1, 8], dtype=torch.long)
pos = torch.rand(6, 3)
batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
energies, forces = model(z, pos, batch)
print(energies)
print(forces)
