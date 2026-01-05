# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from torchmdnet.models.model import load_model
import torch
import numpy as np


def optimize_geometries(model_file_path, z, pos, batch, q, steps=10, **kwargs):
    """Optimize geometries using torch.optim.

    This function performs energy minimization using a TorchMD-Net model
    to calculate potential energy and gradients with torch.optim.LBFGS as
    the optimizer.


    Parameters
    ----------
    model_file_path : str
        Path to the torchmdnet model
    z : torch.Tensor
        Atomic numbers for all atoms in the batch (shape: [N]).
    pos : torch.Tensor
        Initial Cartesian coordinates for the atoms (shape: [N, 3]).
    batch : torch.Tensor
        A mapping of each atom to its respective molecule index in the
        batch (shape: [N]).
    q : torch.Tensor
        Total molecular charges
        (shape: [num_molecules]).
    steps : int, optional
        The number of LBFGS steps to perform. Default: 10. Note that within
        one LBFGS step there are up to 20 iterations (max_iter=20,
        see https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html).
    kwargs : dict, optional
            Additional parameters passed to the torchmdnet model.
    Returns
    -------
    torch.Tensor
        The optimized Cartesian coordinates after the specified number
        of iterations.
    """

    # there are some defaults we need to set
    if "max_num_neighbors" not in kwargs:
        kwargs["max_num_neighbors"] = 64
    if "remove_ref_energy" not in kwargs:
        kwargs["remove_ref_energy"] = True
    model = load_model(model_file_path, derivative=False, **kwargs)
    for parameter in model.parameters():
        parameter.requires_grad = False

    pos = pos.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([pos])

    energies = []

    def closure():
        optimizer.zero_grad()
        energy, _ = model(z, pos, batch, q=q)

        energies.append(energy.detach().cpu().numpy())

        energy = energy.sum()
        energy.backward()

        return energy

    for i in range(steps):
        optimizer.step(closure)

    energies = np.array(energies).squeeze(-1).T

    return pos.detach().cpu().numpy(), energies


# Integration functions copied from TorchMD and modified for batch use
TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR


def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def langevin(vel, gamma, coeff, dt, device):
    csi = torch.randn_like(vel, device=device) * coeff
    vel += -gamma * vel * dt + csi


def batched_kinetic_energy(masses, vel, batch):
    v_sq = torch.sum(vel**2, dim=1)
    E_per_node = 0.5 * masses.squeeze(-1) * v_sq
    n_batch = int(torch.max(batch).item() + 1)
    Ekin = torch.zeros(n_batch, device=vel.device, dtype=vel.dtype).index_add(
        0, batch, E_per_node
    )
    return Ekin


def batched_kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


def kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


class BatchedMLIPIntegrator:
    """Batched Integrator for MLIP systems.

    This class handles the integration of the equations of motion for molecular
    systems using Machine Learning Interatomic Potentials (MLIPs), supporting
    simultaneous simulation of multiple systems in a batched format.

    Modified from https://github.com/torchmd/torchmd

    Parameters
    ----------
    model_file_path : str
        Path to the torchmdnet model
    z : torch.Tensor
        Atomic numbers for all atoms in the batch (shape: [N]).
    pos : torch.Tensor
        Initial Cartesian coordinates for the atoms (shape: [N, 3]).
    masses : torch.Tensor
        Atomic masses for the atoms in the system (shape: [N]).
    batch : torch.Tensor
        A mapping of each atom to its respective molecule index in the
        batch (shape: [N]).
    q : torch.Tensor
        Total molecular charges
        (shape: [num_molecules]).
    timestep : float
        The integration time step in fs.
    device : str, optional
        Default: "cuda"
    temperature : float | None, optional
        Langevin thermostat temperature in Kelvin. Default: None
        If None the integrator will perform NVE Verlet dynamics.
    gamma : float, optional
        Langevin thermostat friction in 1/ps. Default: 1
    kwargs : dict, optional
        Additional parameters passed to the torchmdnet model.
    """

    def __init__(
        self,
        model_file_path,
        z,
        pos,
        masses,
        batch,
        q,
        timestep,
        device="cuda",
        gamma=1.0,
        T=None,
        **kwargs,
    ):

        # load the model

        # there are some defaults we need to set
        if "max_num_neighbors" not in kwargs:
            kwargs["max_num_neighbors"] = 64
        if "remove_ref_energy" not in kwargs:
            kwargs["remove_ref_energy"] = True
        self.model = load_model(model_file_path, derivative=True, **kwargs)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.to(device)
        self.z = z
        self.M = masses.unsqueeze(-1)
        self.pos = pos
        self.batch = batch
        self.q = q
        self.box = None
        self.device = device
        self.dt = timestep / TIMEFACTOR
        gamma = gamma / PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        if T:
            self.vcoeff = torch.sqrt(2.0 * gamma / self.M * BOLTZMAN * T * self.dt).to(
                device
            )
        self.vel = torch.zeros_like(pos)
        self.forces = torch.zeros_like(pos)
        self.n_atoms_per_batch = torch.bincount(batch)
        print(self.n_atoms_per_batch)

    def step(self, niter=1):
        for _ in range(niter):
            _first_VV(self.pos, self.vel, self.forces, self.M, self.dt)

            _pos = self.pos.clone().requires_grad_(True)
            pot, f = self.model(self.z, _pos, self.batch, q=self.q)

            self.forces = f.clone().detach()

            if self.T:
                langevin(self.vel, self.gamma, self.vcoeff, self.dt, self.device)
            _second_VV(self.vel, self.forces, self.M, self.dt)

        Ekin = batched_kinetic_energy(self.M, self.vel, self.batch)
        T = kinetic_to_temp(Ekin, self.n_atoms_per_batch)

        return (
            Ekin.detach().cpu().numpy(),
            pot.detach().squeeze(-1).cpu().numpy(),
            T.detach().cpu().numpy(),
        )


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
