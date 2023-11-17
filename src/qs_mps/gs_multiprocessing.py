import concurrent.futures
import os
from .mps_class import MPS

def ground_state_ising_param(params):
    args_mps = params[0]
    param = params[1]
    chain = MPS(L=args_mps['L'], d=args_mps['d'], model=args_mps['model'], chi=args_mps['chi'], h=param, J=args_mps['J'], eps=args_mps['eps'])
    chain._random_state(seed = 7, chi=args_mps['chi'])
    chain.canonical_form(trunc_chi=False, trunc_tol=True)
    
    energy, entropy = chain.DMRG(trunc_tol=args_mps['trunc_tol'],
                                 trunc_chi=args_mps['trunc_chi'], 
                                 where=args_mps['where'], 
                                 bond=args_mps['bond'])
    chain.save_sites(path=args_mps['path'], precision=args_mps['precision'])
    return energy, entropy

def ground_state_ising_multpr(args_mps, multpr_param, cpu_percentage=90):
    max_workers = int(os.cpu_count()*(cpu_percentage/100)) 
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [[args_mps, param] for param in multpr_param]
        results = executor.map(ground_state_ising_param, args)

    energies = []
    entropies = []
    for result in results:
        energies.append(result[0])
        entropies.append(result[1])
    return energies, entropies

def ground_state_ising(args_mps, multpr, param):
    
    if multpr:
        energies_param, entropies_param = ground_state_ising_multpr(args_mps=args_mps, multpr_param=param)
    else:
        energies_param = []
        entropies_param = []
        for p in param:
            params = [args_mps, p]
            energies, entropies = ground_state_ising_param(params=params)
            energies_param.append(energies)
            entropies_param.append(entropies)

    return energies_param, entropies_param