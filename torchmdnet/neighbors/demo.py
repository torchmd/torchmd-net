import time
import torch as pt
from moleculekit.molecule import Molecule
from torch.utils.benchmark import Timer
from torchmdnet.neighbors import get_neighbor_list

radius = 10

device = "cpu"

files = [ 'alanine_dipeptide.pdb', 'testosterone.pdb', 'chignolin.pdb', 'dhfr.pdb', 'factorIX.pdb', 'stmv.pdb' ]

pdb_file = "alanine_dipeptide.pdb"
pdb_file = "benchmarks/systems/" + pdb_file

print(f"{device}: Reading molecule {pdb_file}")

molecule = Molecule(pdb_file)
positions = pt.tensor(molecule.coords[:,:,0], dtype=pt.float32, device=device)

print(f"{device}: Getting neighbors")

device_positions = positions

start = time.process_time()
result_cpu = get_neighbor_list(device_positions, radius, 500)
print(f"{device}: Took {time.process_time() - start} seconds")
print(f"{device}: Result length: {len(result_cpu[0])}")

device = "cuda"

print(f"{device}: Reading molecule {pdb_file}")

molecule = Molecule(pdb_file)
positions = pt.tensor(molecule.coords[:,:,0], dtype=pt.float32, device=device)

print(f"{device}: Getting neighbors")

device_positions = positions

start = time.process_time()
result_cuda = get_neighbor_list(device_positions, radius, 500)
print(f"{device}: Took {time.process_time() - start} seconds")

print(f"{device}: Result length: {len(result_cuda[0])}")

list_cpu = [tuple if tuple[0] > tuple[1] else (tuple[1], tuple[0]) for tuple in list(zip(result_cpu[0].tolist(), result_cpu[1].tolist()))]
list_cuda = [tuple if tuple[0] > tuple[1] else (tuple[1], tuple[0]) for tuple in list(zip(result_cuda[1].tolist(), result_cuda[0].tolist()))]

list_cpu.sort()
list_cuda.sort()

print(f"Obtained neighbor lists are equivalent: {list_cpu == list_cuda}")
