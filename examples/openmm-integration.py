# This script shows how to use a TorchMD-Net model as a force field in OpenMM
# We will run some simulation steps with OpenMM on chignolin using a pretrained model.

try:
    import openmm
    import openmmtorch
except ImportError:
    raise ImportError("Please install OpenMM and OpenMM-Torch (you can use conda install -c conda-forge openmm openmm-torch)")

import sys
import torch
from openmm.app import PDBFile, StateDataReporter, Simulation
from openmm import Platform, System
from openmm import LangevinMiddleIntegrator
from openmm.unit import *
from torchmdnet.models.model import load_model


# This is a wrapper that links an OpenMM Force with a TorchMD-Net model
class Wrapper(torch.nn.Module):

    def __init__(self, embeddings, model):
        super(Wrapper, self).__init__()
        self.embeddings = embeddings
        # Load a model checkpoint from a previous training run.
        # You can generate the checkpoint using the examples in this folder, for instance:
        # torchmd-train --conf TensorNet-ANI1X.yaml

        # OpenMM will compute the forces by backpropagating the energy,
        # so we can load the model with derivative=False
        # In this particular example I find that the maximum number of neighbors required is around 40
        self.model = load_model(model, derivative=False, max_num_neighbors=40)

    def forward(self, positions):
        # OpenMM works with nanometer positions and kilojoule per mole energies
        # Depending on the model, you might need to convert the units
        positions = positions.to(torch.float32) * 10.0 # nm -> A
        energy = self.model(z=self.embeddings, pos=positions)[0]
        return energy * 96.4916 # eV -> kJ/mol


pdb = PDBFile("../benchmarks/systems/chignolin.pdb")

# Typically models are trained using atomic numbers as embeddings
z = [i.element.atomic_number for i in pdb.topology.atoms()]
z = torch.tensor(z, dtype=torch.long)

model = torch.jit.script(Wrapper(z, "model.ckpt"))
# Create a TorchForce object from the model
torch_force = openmmtorch.TorchForce(model)

system = System()
# Create an OpenMM system and add the TorchForce
for i in range(pdb.topology.getNumAtoms()):
    system.addParticle(1.0)
system.addForce(torch_force)
integrator = LangevinMiddleIntegrator(298.15*kelvin, 1/picosecond, 2*femtosecond)
platform = Platform.getPlatformByName('CPU')
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(StateDataReporter(sys.stdout, 1, step=True, potentialEnergy=True, temperature=True))
simulation.step(10)
