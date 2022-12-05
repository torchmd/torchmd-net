from torchmdnet.priors import D2
import torch as pt
from pytest import mark


# This functions is intentionally not used.
# It is here just to document how the reference values were computed.
def compute_with_psi4(z, pos):
    """
    Compute the D2 term with Psi4

    Arguments
    ---------
    z: atomic_numbers
    pos: atom positions (Å)

    Return
    ------
    energy (Hartree)
    """
    from psi4.core import Molecule
    from psi4.driver import EmpiricalDispersion

    mol = Molecule.from_arrays(elez=z, geom=pos)
    disp = EmpiricalDispersion(name_hint="TPSS-D2")  # Use TPSS, because s_6=1.0
    energy = disp.compute_energy(mol)

    return energy


# The values of `y` are computed with the function above.
# NOTE: Psi4 is not compatible with conda-forge
TEST_CASES = {
    "he": {
        "z": [2],
        "pos": [[0.0, 0.0, 0.0]],
        "batch": [0],
        "y": [0.0],
    },
    "h2": {
        "z": [1, 1],
        "pos": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "batch": [0, 0],
        "y": [-2.396695338419219e-06],
    },
    "h2o_n2_ch4": {
        "z": [1, 8, 1, 7, 7, 6, 1, 1, 1],
        "pos": [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        "batch": [0, 0, 0, 1, 1, 2, 2, 2, 2],
        "y": [-4.7815923481300465e-06, -3.0071055390390565e-06, -7.100283425158895e-05],
    },
}


@mark.parametrize("test_case", TEST_CASES.items())
def test_d2(test_case):
    name, data = test_case

    prior = D2(
        cutoff_distance=10.0,  # Å
        max_num_neighbors=128,
        atomic_number=list(range(100)),
        distance_scale=1e-10,  # Å --> m
        energy_scale=4.35974e-18,  # Hartree --> J
    )

    y_ref = pt.tensor(data["y"])
    z = pt.tensor(data["z"])
    pos = pt.tensor(data["pos"])
    batch = pt.tensor(data["batch"])

    y_init = pt.zeros_like(y_ref)
    y_res = prior.post_reduce(y_init, z, pos, batch, {})

    pt.testing.assert_allclose(y_res, y_ref)
