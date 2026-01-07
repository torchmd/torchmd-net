# This script demonstrates how to do batched minimization of molecules with AceFF torchmd-net models
# It uses TorchMD for the minimizer and integrator
# you can install it with:
# pip install torchmd

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from torchmdnet.models.model import load_model
from torchmd.integrator import Integrator
from torchmd.systems import System
from torchmd.minimizers import minimize_pytorch_bfgs
import torch
import time


# first we define some helper functions
def rdkit_confgen(smiles, N):
    """Standard RDKit confgen method"""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    conformers = rdDistGeom.EmbedMultipleConfs(
        mol, useRandomCoords=True, numConfs=N, numThreads=8
    )

    return mol


def mols_to_batch(mols, device="cuda"):
    """Helper function to convert a group of RDKit molecules into a TorchMD-Net batch.

    This function processes a list of RDKit molecules, handling cases with
    multiple conformers per molecule, and aggregates the data into a one batch

    Parameters
    ----------
    mols : list of rdkit.Chem.rdchem.Mol
        A list containing RDKit molecule objects. Each molecule may
        contain one or more conformers.
    device : str or torch.device, optional
        The device where the resulting tensors will be allocated.
        Default: 'cuda'

    Returns
    -------
    z : torch.Tensor
        Atomic numbers for all atoms across all molecules and conformers
        (shape: [Total_Atoms]).
    pos : torch.Tensor
        Cartesian coordinates for all atoms (shape: [Total_Atoms, 3]).
    batch : torch.Tensor
        Batch index for each atom, mapping it to its respective molecule
        or conformer index (shape: [Total_Atoms]).
    q : torch.Tensor
        Formal or total charges for each system in the batch
        (shape: [Num_Systems]).
    """
    z_list = []
    pos_list = []
    m_list = []
    batch_list = []
    mol_conformer_idx = 0

    charges = []

    for mol in mols:
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        atom_nums = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
        )
        masses = torch.tensor(
            [atom.GetMass() for atom in mol.GetAtoms()], dtype=torch.long
        )
        num_atoms = atom_nums.shape[0]
        for conf in mol.GetConformers():
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)  # (n_atoms, 3)
            z_list.append(atom_nums)
            pos_list.append(pos)
            m_list.append(masses)
            batch_list.append(
                torch.full((num_atoms,), mol_conformer_idx, dtype=torch.long)
            )
            mol_conformer_idx += 1

            charges.append(torch.tensor(total_charge))

    z = torch.cat(z_list, dim=0).to(device)
    pos = torch.cat(pos_list, dim=0).to(device)
    m = torch.cat(m_list, dim=0).to(device)
    batch = torch.cat(batch_list, dim=0).to(device)
    q = torch.stack(charges, dim=0).to(device)
    return z, pos, m, batch, q


def batch_to_mols(pos, batch, mols):
    """Helper function to assign new positions to the RDKit molecules.

    This function updates the Cartesian coordinates of a set of RDKit
    molecules using optimized positions or trajectory data stored in
    tensors.

    Parameters
    ----------
    pos : torch.Tensor
        Cartesian coordinates to be assigned to the molecules
        (shape: [N, 3])
    batch : torch.Tensor
        Mapping of each atom to its respective molecule index (shape: [N]).
    mols : list of rdkit.Chem.rdchem.Mol
        The original list of RDKit molecules that will be updated with
        the new coordinate data.

    Returns
    -------
    list of rdkit.Chem.rdchem.Mol
        The list of RDKit molecules with updated conformers reflecting
        the provided positions.
    """

    from rdkit.Geometry import Point3D

    counter = 0
    for mol in mols:
        for conf in mol.GetConformers():
            indexes = batch.detach().cpu().numpy() == counter
            new_pos = pos[indexes]

            for i in range(mol.GetNumAtoms()):
                x, y, z = new_pos[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

            counter += 1

    return mols


class Calculator:
    """A wrapper class to use a torchmd-net model with torchmd"""

    def __init__(
        self, model_file_path, z, pos, batch, q, device="cuda", derivative=False
    ):
        """Initializes the Calculator with a specific model and system. z,pos,batch,q should be in torchmd-net style batches
        Args:
            model_file_path (str): Path to the torchmd-net model
            z (torch.Tensor): Atomic numbers, shape [N]
            pos (torch.Tensor): Initial coordinates of the atoms, shape [N,3].
            batch (torch.Tensor): Batch index tensor for the atoms, shape [N]
            q (torch.Tensor): Molecule charges, shape [num_molecules]
            device (str, optional): Hardware device to run inference on. Defaults to "cuda".
            derivative (bool, optional): If True, enables gradient calculation
                for forces. Defaults to False.
        """

        # there are some defaults we need to set
        # you can pass the standard extra args to torchmd-net models here
        kwargs = {"max_num_neighbors": 64, "remove_ref_energy": True}
        self.derivative = derivative
        model = load_model(model_file_path, derivative=derivative, **kwargs)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.to(device)

        self.model = model
        self.z = z
        self.pos = pos
        self.batch = batch
        self.q = q

    def compute(self, pos, box, forces_tensor, toNumpy=False):
        energies = []
        for i in range(
            pos.shape[0]
        ):  # Iterate over system replicas, in this example there will be 1 system that contains multiple molecules in a batch
            energy, forces = self.model(self.z, pos[i], self.batch, q=self.q)
            energies.append(energy)
            if self.derivative:
                forces_tensor[i].copy_(forces.detach())
        return energies


if __name__ == "__main__":

    # get the AceFF2.0 model
    from huggingface_hub import hf_hub_download

    model_file = hf_hub_download(
        repo_id="Acellera/AceFF-2.0", filename="aceff_v2.0.ckpt"
    )

    # Example molecules
    smiles_list = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)OC1=CC=CC=C1C(=O)O"]

    # Number of conformers to generate
    N = 10

    # Generate conformers with RDKit
    mols = [rdkit_confgen(smiles, N) for smiles in smiles_list]

    # convert list of RDKit molecules to a torchmd-net style batch
    device = "cuda"
    z, pos, m, batch, q = mols_to_batch(mols, device=device)

    # create the TorchMD system
    # here we make one system with a batch of multiple molecules.
    system = System(len(z), 1, precision=torch.float, device=device)
    system.set_positions(pos)
    system.set_masses(m)

    # create the TorchMD calculator, for this one we set derivative=False because gradient calculation will be handled by the minimizer
    calculator = Calculator(
        model_file, z, pos, batch, q, device=device, derivative=False
    )

    # Minimize the energy with the batched pytorch minimizer
    # for larger molecules you may need to increase the steps.
    # Note that each "step" performs multiple torch.optim.lbfgs iterations (max_iter=20)
    energy_trajectories = minimize_pytorch_bfgs(system, calculator, steps=2)[0]

    # the function returns the energy trajectories in shape [n_conformers, n_iterations]
    print("Minimized energies:")
    print(energy_trajectories.shape)
    print(energy_trajectories[:, -1])

    # the minimized coordinates can be taken from the system
    minimized_pos = system.pos.squeeze(0)  # torchmd will add a leading dim

    # convert back to RDKit
    mols = batch_to_mols(minimized_pos, batch, mols)

    # plot the energy trajectories to check convergence
    plt.figure()
    for i in range(0, N):
        plt.plot(energy_trajectories[i, :])
    plt.ylabel("Energy (eV)")
    plt.xlabel("Minimization iteration")
    plt.savefig("energies_mol_1.png")

    plt.figure()
    for i in range(N, 2 * N):
        plt.plot(energy_trajectories[i, :])
    plt.ylabel("Energy (eV)")
    plt.xlabel("Minimization iteration")
    plt.savefig("energies_mol_2.png")

    # now we can run batched MD

    # create the integrator
    langevin_temperature = 300  # K
    langevin_gamma = 1.0  # 1/ps
    timestep = 1  # fs

    # create a calculator again, for this one we need derivative=True
    calculator = Calculator(
        model_file, z, pos, batch, q, device=device, derivative=True
    )
    # create the integrator
    integrator = Integrator(
        system,
        calculator,
        timestep,
        device,
        T=langevin_temperature,
        gamma=langevin_gamma,
        batch=batch,
    )
    system.set_positions(pos)

    # run the MD, 10 iterations of 100 steps
    inner_steps = 100
    for i in range(10):
        t1 = time.perf_counter()
        Ekin, pot, T = integrator.step(inner_steps)
        t2 = time.perf_counter()
        print("step:", (i + 1) * inner_steps)
        print("energies:", pot[0][:, 0])
        print("T:", T)
        print(f"time per step: {(t2-t1)/inner_steps*1000} ms")
