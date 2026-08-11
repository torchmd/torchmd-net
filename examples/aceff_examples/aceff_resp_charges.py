# This script shows how to load the AceFF-2-RESP-1 checkpoint and predict
# per-atom RESP partial charges for a molecule.
import torch

# The AceFF models are available from HuggingFace under apache2.0 license
from huggingface_hub import hf_hub_download

model_file_path = hf_hub_download(
    repo_id="Acellera/AceFF-2-RESP-1", filename="aceff-2-resp-1.ckpt"
)

from torchmdnet.models.model import load_model

# model="tensornet2_resp" selects the RESP architecture (the checkpoint stores "tensornet2")
model = load_model(model_file_path, model="tensornet2_resp", derivative=True)

# single molecule (water: O, H, H)
z = torch.tensor([8, 1, 1], dtype=torch.long)
pos = torch.tensor([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
q = torch.zeros(1)  # total molecular charge

energy, forces, charges = model(z=z, pos=pos, q=q)
print("RESP charges:", charges.reshape(-1))   # per-atom, sums to the total charge
