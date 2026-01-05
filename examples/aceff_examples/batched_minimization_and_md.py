# This script demonstrates how to do batched minimization of molecules with AceFF torchmd-net models

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
import torch
from torchmdnet.examples_utils import (
    optimize_geometries,
    batch_to_mols,
    mols_to_batch,
    BatchedMLIPIntegrator,
)
import time


def rdkit_confgen(smiles, N):
    """Standard RDKit confgen method"""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    conformers = rdDistGeom.EmbedMultipleConfs(
        mol, useRandomCoords=True, numConfs=N, numThreads=8
    )

    return mol


if __name__ == "__main__":

    # get the AceFF2.0 model
    from huggingface_hub import hf_hub_download

    model_file_path = hf_hub_download(
        repo_id="Acellera/AceFF-2.0", filename="aceff_v2.0.ckpt"
    )

    # Example molecules
    smiles_list = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)OC1=CC=CC=C1C(=O)O"]

    # Number of conformers to generate
    N = 10

    # Generate conformers with RDKit
    mols = [rdkit_confgen(smiles, N) for smiles in smiles_list]

    # convert list of RDKit molecules to a torchmdnet style batch
    device = "cuda"
    z, pos, m, batch, q = mols_to_batch(mols, device=device)

    # minimize
    minimized_pos, energy_trajectories = optimize_geometries(
        model_file_path, z, pos, batch, q, device=device
    )

    print("Minimized energies:")
    energy_trajectories[:, -1]

    # Minimized_pos are the minimized coordinates, shape [n_atoms, 3]
    # energy_trajectories are the energy per iteration for each conformer, shape [n_conformers, n_iterations]

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
    integrator = BatchedMLIPIntegrator(
        model_file_path,
        z,
        pos,
        m,
        batch,
        q,
        device="cuda",
        timestep=timestep,
        gamma=langevin_gamma,
        T=langevin_temperature,
    )

    # run the MD, 10 iterations of 100 steps
    inner_steps = 100
    for i in range(10):
        t1 = time.perf_counter()
        Ekin, pot, T = integrator.step(inner_steps)
        t2 = time.perf_counter()
        print("step:", (i + 1) * inner_steps)
        print("energies:", pot)
        print("T:", T)
        print(f"time per step: {(t2-t1)/inner_steps*1000} ms")
