import concurrent.futures
import os
from qs_mps.mps_class import MPS
from qs_mps.utils import tensor_shapes

def ground_state_ANNNI_param(params):
    args_mps = params[0]
    h = params[1][0]
    k = params[1][1]
    print(h,k, type(h), type(k))
    chain = MPS(
        L=args_mps["L"],
        d=args_mps["d"],
        model=args_mps["model"],
        chi=args_mps["chi"],
        h=h,
        J2=k,
        J=1
    )
    save = args_mps["save"]
    precision = args_mps["precision"]
    chain._random_state(seed=3, chi=args_mps["chi"], type_shape=args_mps["type_shape"])
    tensor_shapes(chain.sites)
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
        chain.save_sites(args_mps["path"], args_mps["precision"], args_mps["charges_x"], args_mps["charges_y"])
    return energy, entropy


def ground_state_ANNNI_multpr(args_mps, multpr_param, cpu_percentage=90):
    max_workers = int(os.cpu_count() * (cpu_percentage / 100))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [[args_mps, [h,k]] for h in multpr_param[0] for k in multpr_param[1]]
        results = executor.map(ground_state_ANNNI_param, args)

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


def ground_state_ANNNI(args_mps, multpr, param):
    if multpr:
        energies_param, entropies_param = ground_state_ANNNI_multpr(
            args_mps=args_mps, multpr_param=param
        )
    else:
        energies_param = []
        entropies_param = []
        for h in param[0]:
            for k in param[1]:
                params = [args_mps, [h,k]]
                energy, entropy = ground_state_ANNNI_param(params=params)
                energies_param.append(energy[-1])
                entropies_param.append(entropy)

    return energies_param, entropies_param
