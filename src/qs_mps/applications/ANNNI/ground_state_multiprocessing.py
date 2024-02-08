import concurrent.futures
import os
from qs_mps.mps_class import MPS
from qs_mps.utils import tensor_shapes
import numpy as np

def ground_state_ANNNI_param(params):
    args_mps = params[0]
    h = params[1][0]
    k = params[1][1]
    chain = MPS(
        L=args_mps["L"],
        d=args_mps["d"],
        model=args_mps["model"],
        chi=args_mps["chi"],
        h=h,
        k=k,
        J=1
    )
    save = args_mps["save"]
    precision = args_mps["precision"]
    chain._random_state(seed=3, chi=args_mps["chi"], type_shape=args_mps["type_shape"])
    # tensor_shapes(chain.sites)
    chain.canonical_form(trunc_chi=True, trunc_tol=False)
    
    energy, entropy, schmidt_vals = chain.DMRG(
        trunc_tol=args_mps["trunc_tol"],
        trunc_chi=args_mps["trunc_chi"],
        where=args_mps["where"],
        bond=args_mps["bond"],
    )
    print(f"energy of h:{h:.{precision}f}, k:{k:.{precision}f} is:\n {energy}")
    print(f"Schmidt values in the middle of the chain:\n {schmidt_vals}")

    if save:
        chain.save_sites(args_mps["path"], args_mps["precision"])
    return energy, entropy, schmidt_vals


def ground_state_ANNNI_multpr(args_mps, multpr_param, cpu_percentage=90):
    max_workers = int(os.cpu_count() * (cpu_percentage / 100))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [[args_mps, [h,k]] for h in multpr_param[0] for k in multpr_param[1]]
        results = executor.map(ground_state_ANNNI_param, args)

    energies = []
    entropies = []
    precision = args_mps["precision"]
    i = 0
    j = 0
    for result in results:

        print(f"energy of h:{multpr_param[0][i]:.{precision}f}, k:{multpr_param[1][j]:.{precision}f} is:\n {result[0][-1]}")
        print(f"Schmidt values in the middle of the chain:\n {result[2]}")

        energies.append(result[0])
        entropies.append(result[1])
        if j == (len(multpr_param[1])-1):
            i += 1
            j = -1
        j += 1
        

    return energies, entropies


def ground_state_ANNNI(args_mps, multpr, param):
    if multpr:
        energies_param, entropies_param = ground_state_ANNNI_multpr(
            args_mps=args_mps, multpr_param=param
        )
    else:
        energies_param = []
        entropies_param = []
        schmidt_vals_param = []
        for h in param[0]:
            energies_h = []
            entropies_h = []
            schmidt_vals_h = []
            for k in param[1]:
                # if h == 0 and k > 0.48:
                params = [args_mps, [h,k]]
                energy, entropy, schmidt_vals = ground_state_ANNNI_param(params=params)
                energies_h.append(energy[-1])
                entropies_h.append(entropy)
                schmidt_vals_h.append(schmidt_vals)
            energies_param.append(energies_h)
            entropies_param.append(entropies_h)
            schmidt_vals_param.append(schmidt_vals_h)

    return energies_param, entropies_param, schmidt_vals_param
