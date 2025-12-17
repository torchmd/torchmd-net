# This script shows how to use the Atomic Simulation Environment Calculator (ASE)
# inferface of TorchMD-Net with an AceFF model 
import time
import sys
import ase
from ase.io import read
from torchmdnet.calculators import TMDNETCalculator

# The AceFF-1.0 model is available from HuggingFace under apache2.0 license 
from huggingface_hub import hf_hub_download

model_file_path = hf_hub_download(
    repo_id="Acellera/AceFF-1.0",
    filename="aceff_v1.0.ckpt"
)


# We create the ASE calculator by supplying the path to the model and specifying the device and dtype
calc  = TMDNETCalculator(model_file_path, device='cuda')
atoms = read('caffeine.pdb')
print(atoms)

atoms.calc = calc

# The total molecular charge must be set 
atoms.info['charge'] = 0

energy = atoms.get_potential_energy()
print(energy)
forces = atoms.get_forces()
print(forces)


# We can use all the normal ASE methods

# Energy minimization
from ase.optimize import LBFGS

# displace the atoms to a high energy
atoms.rattle(0.5)
print("Initial energy:", atoms.get_potential_energy())

dyn = LBFGS(atoms)
dyn.run()
print("Minimized energy:", atoms.get_potential_energy())



# Molecular dynamics
from ase import units
from ase.md.langevin import Langevin
from ase.md import MDLogger


# setup MD
temperature_K: float = 300
timestep: float = 1.0 * units.fs
friction: float = 0.01 / units.fs
traj_interval: int = 1000
log_interval: int   = 100
nsteps: int = 1000

dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=friction)
dyn.attach(
    lambda: ase.io.write("traj.xyz", atoms, append=True), interval=traj_interval
)
dyn.attach(MDLogger(dyn, atoms,  sys.stdout), interval=log_interval)


# Run the dynamics
t1 = time.perf_counter()
dyn.run(steps=nsteps)
t2 = time.perf_counter()

print(f"Completed MD in {t2 - t1:.1f} s ({(t2 - t1)*1000 / nsteps:.3f} ms/step)")



# Now we can do the same but enabling torch.compile for increased speed
calc  = TMDNETCalculator(model_file_path, device='cuda', compile=True)

atoms.calc = calc

# Single point calcuation to trigger compile
atoms.get_potential_energy()

# Run more dynamics
t1 = time.perf_counter()
dyn.run(steps=nsteps)
t2 = time.perf_counter()

print(f"Completed MD in {t2 - t1:.1f} s ({(t2 - t1)*1000 / nsteps:.3f} ms/step)")
