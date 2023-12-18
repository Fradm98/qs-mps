import concurrent.futures
import os
from qs_mps.mps_class import MPS
from qs_mps.applications.Z2.exact_hamiltonian import H_Z2_gauss


def ground_state_Z2_exact_param(params):
    args_lattice = params[0]
    param = params[1]
    Z2 = H_Z2_gauss(L=args_lattice["L"], l=args_lattice["l"], model=args_lattice["model"], U=args_lattice["U"], lamb=param)
    e, v = Z2.diagonalize(v0=args_lattice["v0"], path=args_lattice["path"], save=args_lattice["save"], precision=args_lattice["precision"])
    return e, v

def ground_state_Z2_exact(args_lattice, param):
    energies_param = []
    for p in param:
        params = [args_lattice, p]
        energy, vectors = ground_state_Z2_exact_param(params=params)
        energies_param.append(energy)
        v0 = vectors[:,0]
        args_lattice["v0"] = v0

    return energies_param


def ground_state_Z2_param(params):
    args_mps = params[0]
    param = params[1]
    ladder = MPS(
        L=args_mps["L"],
        d=args_mps["d"],
        model=args_mps["model"],
        chi=args_mps["chi"],
        h=param,
    )
    save = args_mps["save"]
    if ladder.model == "Z2_dual":
        ladder.L = ladder.L - 1
    ladder._random_state(seed=3, chi=args_mps["chi"], type_shape=args_mps["type_shape"])
    ladder.canonical_form(trunc_chi=True, trunc_tol=False)
    energy, entropy = ladder.DMRG(
        trunc_tol=args_mps["trunc_tol"],
        trunc_chi=args_mps["trunc_chi"],
        where=args_mps["where"],
    )

    if save:
        ladder.save_sites(args_mps["path"], args_mps["precision"])
    return energy, entropy


def ground_state_Z2_multpr(args_mps, multpr_param, cpu_percentage=90):
    max_workers = int(os.cpu_count() * (cpu_percentage / 100))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [[args_mps, param] for param in multpr_param]
        results = executor.map(ground_state_Z2_param, args)

    energies = []
    entropies = []
    i = 0
    for result in results:
        print(f"enegy of h:{multpr_param[i]} is:\n {result[0][-1]}")
        energies.append(result[0])
        entropies.append(result[1])
        i += 1
    return energies, entropies


def ground_state_Z2(args_mps, multpr, param):
    if multpr:
        energies_param, entropies_param = ground_state_Z2_multpr(
            args_mps=args_mps, multpr_param=param
        )
    else:
        energies_param = []
        entropies_param = []
        for p in param:
            params = [args_mps, p]
            energy, entropy = ground_state_Z2_param(params=params)
            energies_param.append(energy[-1])
            entropies_param.append(entropy)

    return energies_param, entropies_param


import numpy as np
from qs_mps.utils import get_precision
chis = [2, 4]
h_i = 0
h_f = 10
npoints = 100
L = 3
l = 2
d = int(2**l)
type_shape = "trapezoidal"
model = "Z2_dual"
path_tensor = "D:/code/projects/1_Z2"
save = False
multpr = False

interval = np.linspace(h_i, h_f, npoints)
num = (h_f - h_i) / npoints
precision = get_precision(num)

for chi in chis:  # L // 2 + 1
    args_mps = {
        "L": L,
        "d": d,
        "chi": chi,
        "type_shape": type_shape,
        "model": model,
        "trunc_tol": False,
        "trunc_chi": True,
        "where": L // 2,
        "path": path_tensor,
        "save": save,
        "precision": precision,
    }
    if __name__ == "__main__":
        energy_chi, entropy_chi = ground_state_Z2(
            args_mps=args_mps, multpr=multpr, param=interval
        )
