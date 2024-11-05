import concurrent.futures
import os
from qs_mps.mps_class import MPS
from qs_mps.applications.Z2.exact_hamiltonian import H_Z2_gauss
from qs_mps.utils import tensor_shapes
import signal
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Define a function to handle the timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Algorithm took too long to execute")

# Set the signal handler
signal.signal(signal.SIGALRM, timeout_handler)

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
        bc=args_mps["bc"],
        h=param,
    )
    save = args_mps["save"]
    precision = args_mps["precision"]
    if ladder.model == "Z2_dual":
        if args_mps["sector"] != "vacuum_sector":
            ladder.Z2.add_charges(rows=args_mps["charges_x"], columns=args_mps["charges_y"])
            print(ladder.Z2.charges)
    if args_mps["guess"] == []:
        ladder._random_state(seed=3, chi=args_mps["chi"], type_shape=args_mps["type_shape"])
        tensor_shapes(ladder.sites)
        ladder.canonical_form(trunc_chi=True, trunc_tol=False)
        if ladder.bc == "pbc":
            a = np.zeros((1,2))
            a[0,0] = 1
            extra_ancillary_site = a.reshape((1,2,1))
            ladder.sites.append(extra_ancillary_site)
            ladder.L = len(ladder.sites)
    else:
        ladder.sites = args_mps["guess"].copy()
        ladder.enlarge_chi()

    energy, entropy, schmidt_vals, t_dmrg = ladder.DMRG(
        trunc_tol=args_mps["trunc_tol"],
        trunc_chi=args_mps["trunc_chi"],
        where=args_mps["where"],
        bond=args_mps["bond"],
        n_sweeps=args_mps["n_sweeps"],
        conv_tol=args_mps["conv_tol"]
    )

    print(f"energy of h:{param:.{precision}f}, L:{ladder.L} is:\n {energy}")
    print(f"Schmidt values in the middle of the chain:\n {schmidt_vals}")
    t_final = np.sum(t_dmrg)
    if t_final < 60:
        t_unit = "sec(s)"
    elif t_final > 60 and t_final < 3600:
        t_unit = "min(s)"
        t_final = t_final/60
    elif t_final > 3600:
        t_unit = "hour(s)"
        t_final = t_final/3600

    print(f"time of the whole search for h={param:.{precision}f} is: {t_final} {t_unit}")
    
    if not args_mps["training"]:
        energy = energy[-1]

    if save:
        if ladder.bc == "pbc":
            ladder.sites.pop()
        ladder.L = len(ladder.sites)
        ladder.save_sites(args_mps["path"], args_mps["precision"], args_mps["charges_x"], args_mps["charges_y"])
    # args_mps["guess"] = ladder.sites.copy()
    return energy, entropy, schmidt_vals, t_dmrg


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


def ground_state_Z2(args_mps, multpr, param, reps: int=3):
    if multpr:
        energies_param, entropies_param = ground_state_Z2_multpr(
            args_mps=args_mps, multpr_param=param
        )
    else:
        energies_param = []
        entropies_param = []
        schmidt_vals_param = []
        time_param = []
        threshold = 5
        slack = 2
        execution_times = []

        for p in param:
            params = [args_mps, p]
            for attempt in range(reps):
                
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(ground_state_Z2_param, params=params)
                    try:
                        # Attempt to execute within the threshold time
                        results = future.result(timeout=threshold)
                        energy, entropy, schmidt_vals, t_dmrg = results
                        print(f"Run for parameter: {p:.2f} attempt: {attempt} completed in {t_dmrg:.2f}s within threshold.")
                        execution_times.append(t_dmrg)
                        energies_param.append(energy)
                        entropies_param.append(entropy)
                        schmidt_vals_param.append(schmidt_vals)
                        time_param.append(t_dmrg)
                        break  # Exit retry loop since run was successful

                    except TimeoutError:
                        print(f"Run for parameter: {p:.2f} attempt: {attempt} exceeded threshold of {threshold:.2f}s. Retrying with random state...")

                    # Update parameters here as needed before retrying
                    args_mps["guess"] = []

            # Update the threshold based on the average time with slack
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                threshold = avg_time * slack
                print(f"New threshold updated to {threshold:.2f}s")


        # for p in param:
        #     params = [args_mps, p]
        #     # Set the timeout period (in seconds)
        #     timeout_secs = threshold # You can change this value according to your requirement
        #     # Set the alarm
        #     signal.alarm(int(timeout_secs+1))
        #     print(f"New timeout seconds: {int(timeout_secs+1)}")
        #     try:
        #         print("Running with guess state:")
        #         energy, entropy, schmidt_vals, t_dmrg = ground_state_Z2_param(params=params)

        #     except TimeoutError as e:
        #         print(e)
        #         args_mps["guess"] = []
        #         print("Running with random state:")
        #         energy, entropy, schmidt_vals, t_dmrg = ground_state_Z2_param(params=params)
        #     else:
        #         # Cancel the alarm if the algorithm finishes before the timeout
        #         signal.alarm(0)

        #     time_param.append(t_dmrg)
        #     t_mean = np.mean(time_param)
        #     t_std = np.std(time_param)
        #     threshold = t_mean + 3*t_std
        #     energies_param.append(energy)
        #     entropies_param.append(entropy)
        #     schmidt_vals_param.append(schmidt_vals)
        #     time_param.append(t_dmrg)


    return energies_param, entropies_param, schmidt_vals_param, time_param
