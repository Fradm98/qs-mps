import concurrent.futures
import os
from qs_mps.mps_class import MPS
from qs_mps.applications.Z2.exact_hamiltonian import H_Z2_gauss
from qs_mps.utils import tensor_shapes

def ground_state_Z2_exact_param(params):
    args_lattice = params[0]
    param = params[1]
    Z2 = H_Z2_gauss(L=args_lattice["L"], l=args_lattice["l"], model=args_lattice["model"], U=args_lattice["U"], lamb=param)
    if len(args_lattice["charges_x"]) > 0:
        # print(Z2.charges)
        Z2.add_charges(rows=args_lattice["charges_x"], columns=args_lattice["charges_y"])
        # print(Z2.charges)
        Z2._define_sector()
        print("before diag")
    e, v = Z2.diagonalize(v0=args_lattice["v0"], path=args_lattice["path"], save=args_lattice["save"], precision=args_lattice["precision"], cx=args_lattice["charges_x"], cy=args_lattice["charges_y"], sparse=args_lattice["sparse"])
    return e, v

def ground_state_Z2_exact(args_lattice, param):
    energies_param = []
    prec = args_lattice["precision"]
    for p in param:
        print(f"computing ground state for param: {p:.{prec}f}")
        params = [args_lattice, p]
        print(args_lattice["v0"])
        energy, vectors = ground_state_Z2_exact_param(params=params)
        energies_param.append(energy)
        v0 = vectors[:,0]
        args_lattice["v0"] = v0

    return energies_param

def ground_state_Z2_exact_test(args_lattice, param):
    energies_param = []
    prec = args_lattice["precision"]
    for p in param:
        print(f"computing ground state for param: {p:.{prec}f}")
        Z2 = H_Z2_gauss(L=args_lattice["L"], l=args_lattice["l"], model=args_lattice["model"], U=args_lattice["U"], lamb=p)
        if len(args_lattice["charges_x"]) > 0:
            print("adding charges")
            Z2.add_charges(rows=args_lattice["charges_x"], columns=args_lattice["charges_y"])
            Z2._define_sector()       
        energy, vectors = Z2.diagonalize(v0=args_lattice["v0"], sparse=args_lattice["sparse"], path=args_lattice["path"], save=args_lattice["save"], precision=args_lattice["precision"], cx=args_lattice["charges_x"], cy=args_lattice["charges_y"])
        energies_param.append(energy)
        v0 = vectors[:,0]
        args_lattice["v0"] = v0
        print(args_lattice["v0"])

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
    precision = args_mps["precision"]
    if ladder.model == "Z2_dual":
        ladder.L = ladder.L - 1
        if args_mps["sector"] != "vacuum_sector":
            ladder.Z2.add_charges(rows=args_mps["charges_x"], columns=args_mps["charges_y"])
            print(ladder.Z2.charges)
    ladder._random_state(seed=3, chi=args_mps["chi"], type_shape=args_mps["type_shape"])
    tensor_shapes(ladder.sites)
    ladder.canonical_form(trunc_chi=True, trunc_tol=False)
    
    energy, entropy, schmidt_vals = ladder.DMRG(
        trunc_tol=args_mps["trunc_tol"],
        trunc_chi=args_mps["trunc_chi"],
        where=args_mps["where"],
        bond=args_mps["bond"],
        n_sweeps=args_mps["n_sweeps"],
        conv_tol=args_mps["conv_tol"]
    )
    print(f"energy of h:{param:.{precision}f}, L:{ladder.L} is:\n {energy}")
    print(f"Schmidt values in the middle of the chain:\n {schmidt_vals}")

    if save:
        ladder.save_sites(args_mps["path"], args_mps["precision"], args_mps["charges_x"], args_mps["charges_y"])
    return energy, entropy, schmidt_vals


def ground_state_Z2_multpr(args_mps, multpr_param, cpu_percentage=90):
    max_workers = int(os.cpu_count() * (cpu_percentage / 100))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [[args_mps, param] for param in multpr_param]
        results = executor.map(ground_state_Z2_param, args)

    energies = []
    entropies = []
    i = 0
    for result in results:
        print(f"energy of h:{multpr_param[i]} is:\n {result[0][-1]}")
        energies.append(result[0])
        print(f"entropies of h:{multpr_param[i]} is:\n {result[1]}")
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
        schmidt_vals_param = []
        for p in param:
            params = [args_mps, p]
            energy, entropy, schmidt_vals = ground_state_Z2_param(params=params)
            energies_param.append(energy[-1])
            entropies_param.append(entropy)
            schmidt_vals_param.append(schmidt_vals)

    return energies_param, entropies_param, schmidt_vals_param
