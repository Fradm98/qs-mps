import concurrent.futures
import os
from qs_mps.mps_class import MPS

def ground_state_Z2_param(params):
    args_mps = params[0]
    param = params[1]
    ladder = MPS(L=args_mps['L'], d=args_mps['d'], model=args_mps['model'], chi=args_mps['chi'], charges=args_mps['charges'], h=param)
    ladder._random_state(seed = 7, chi=args_mps['chi'])
    ladder.canonical_form()
    energy = ladder.DMRG(trunc_tol=args_mps['trunc_tol'],trunc_chi=args_mps['trunc_chi'])
    ladder.save_sites("/Users/fradm98/Desktop/mps/tests/results/tensor_data")
    return energy

def ground_state_Z2_multpr(args_mps, multpr_param, cpu_percentage=90):
    max_workers = int(os.cpu_count()*(cpu_percentage/100)) 
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [[args_mps, param] for param in multpr_param]
        results = executor.map(ground_state_Z2_param, args)

    energies = []
    i = 0
    for result in results:
        print(f"enegy of h:{multpr_param[i]} is:\n {result[-1]}")
        energies.append(result[-1])
        i += 1
    return energies

def ground_state_Z2(args_mps, multpr, param):
    
    if multpr:
        energies_param = ground_state_Z2_multpr(args_mps=args_mps, multpr_param=param)
    else:
        energies_param = []
        for p in param:
            params = [args_mps, p]
            energy = ground_state_Z2_param(params=params)
            energies_param.append(energy[-1])

    return energies_param