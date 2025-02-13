import concurrent.futures
import multiprocessing
import numpy as np
import datetime as dt
from qs_mps.mps_class import MPS
from qs_mps.utils import tensor_shapes, get_precision
import os
from threading import Event


def ground_state_Z2_param(params):
    args_mps = params[0]
    param = params[1]
    ladder = MPS(
        L=args_mps["L"],
        d=args_mps["d"],
        model=args_mps["model"],
        chi=args_mps["chi"],
        bc=args_mps["bc"],
        cc=args_mps["cc"],
        h=param,
    )
    chi = args_mps["chi"]
    save = args_mps["save"]
    precision = args_mps["precision"]
    if ladder.model == "Z2_dual":
        if args_mps["sector"] != "vacuum_sector":
            ladder.Z2.add_charges(
                rows=args_mps["charges_x"], columns=args_mps["charges_y"]
            )
            ladder.Z2._define_sector()
    if args_mps["guess"] == []:
        print("Running with random state")
        ladder._random_state(
            seed=3, chi=args_mps["chi"], type_shape=args_mps["type_shape"]
        )
        ladder.canonical_form(trunc_chi=True, trunc_tol=False)
    else:
        print("Running with guess state")
        ladder.sites = args_mps["guess"].copy()
        ladder.enlarge_chi()

    if ladder.bc == "pbc":
        a = np.zeros((1, 2))
        a[0, 0] = 1
        extra_ancillary_site = a.reshape((1, 2, 1))
        ladder.sites.append(extra_ancillary_site)
        ladder.L = len(ladder.sites)

    energy, entropy, schmidt_vals, t_dmrg = ladder.DMRG(
        trunc_tol=args_mps["trunc_tol"],
        trunc_chi=args_mps["trunc_chi"],
        where=args_mps["where"],
        bond=args_mps["bond"],
        n_sweeps=args_mps["n_sweeps"],
        conv_tol=args_mps["conv_tol"],
    )
    t_final = np.sum(t_dmrg)
    t_final_gen = dt.timedelta(seconds=t_final)
    print(
        f"time of the whole search for h={param:.{precision}f}, chi={chi} is: {t_final_gen} in date {dt.datetime.now()}"
    )

    if not args_mps["training"]:
        energy = energy[-1]

    if save:
        if ladder.bc == "pbc":
            # ladder.sites.pop()
            ladder.L = len(ladder.sites) - 1
        ladder.save_sites(
            args_mps["path"],
            args_mps["precision"],
            args_mps["charges_x"],
            args_mps["charges_y"],
        )
    # new_guess = ladder.sites.copy()

    return energy, entropy, schmidt_vals, t_dmrg
    # return energy, entropy, schmidt_vals, t_dmrg, new_guess


def run_with_timeout(func, args, timeout):
    workers = os.cpu_count()
    with multiprocessing.Pool(processes=int(workers * 0.8)) as pool:
        print(f"\n  --- Using {int(workers*0.8)} processes ---\n")
        result = pool.apply_async(func, args)
        try:
            results = result.get(timeout=timeout)
            pool.terminate()
            pool.join()
            return results
        except multiprocessing.TimeoutError:
            print(
                f"\n## TimeoutError: Algorithm exceeded {timeout} seconds in date {dt.datetime.now()} ##\n"
            )
            pool.terminate()  # Forcefully terminate the worker
            pool.join()
            return None  # Return None or any indicator of failure


# def ground_state_Z2(args_mps, interval, reps=3):
#     ene_tot = []
#     ent_tot = []
#     sm_tot = []
#     t_tot = []
#     timeout = 10000
#     slack = 3
#     precision = args_mps["precision"]
#     params_not_found = []
#     for p in interval:
#         count_attempts = 0
#         print(f"\n*** Starting attempts in {dt.datetime.now()}\n")
#         for attempt in range(reps):
#             params = (args_mps, p)
#             result = run_with_timeout(ground_state_Z2_param, (params,), timeout)
#             if result is None:
#                 print(f"Computation for h={params[1]:.{precision}f} timed out and was terminated in date {dt.datetime.now()}")
#                 args_mps["guess"] = []
#                 timeout = timeout * slack
#                 continue
#             else:
#                 energy, entropy, sm, tdmrg, new_guess = result
#                 ene_tot.append(energy)
#                 ent_tot.append(entropy)
#                 sm_tot.append(sm)
#                 t_tot.append(tdmrg)
#                 args_mps["guess"] = new_guess.copy()
#                 count_attempts = 1
#                 break

#         if count_attempts == 0:
#             print(f"h={params[1]:.{precision}f}")
#             params_not_found.append(params[1])
#         if t_tot:
#             avg_time = sum(t_tot) / len(t_tot)
#             timeout = avg_time * slack
#             print(f"New timeout updated to {timeout:.2f}s in date {dt.datetime.now()}")

#         print(f"\n*** Completed computation in date {dt.datetime.now()} for h={params[1]:.{precision}f}\n")
#     print(f"Parameters not found are {len(params_not_found)}:\n{params_not_found}")

#     return ene_tot, ent_tot, sm_tot, t_tot

# import concurrent.futures
# import os
# import datetime as dt
# from qs_mps.mps_class import MPS
# from qs_mps.applications.Z2.exact_hamiltonian import H_Z2_gauss
# from qs_mps.utils import tensor_shapes
# import signal
# import time
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, TimeoutError

# # # Define a function to handle the timeout
# # def timeout_handler(signum, frame):
# #     raise TimeoutError("Algorithm took too long to execute")

# # # Set the signal handler
# # signal.signal(signal.SIGALRM, timeout_handler)

# def ground_state_Z2_exact_param(params):
#     args_lattice = params[0]
#     param = params[1]
#     Z2 = H_Z2_gauss(L=args_lattice["L"], l=args_lattice["l"], model=args_lattice["model"], U=args_lattice["U"], lamb=param)
#     if len(args_lattice["charges_x"]) > 0:
#         # print(Z2.charges)
#         Z2.add_charges(rows=args_lattice["charges_x"], columns=args_lattice["charges_y"])
#         # print(Z2.charges)
#         Z2._define_sector()
#         print("before diag")
#     e, v = Z2.diagonalize(v0=args_lattice["v0"], path=args_lattice["path"], save=args_lattice["save"], precision=args_lattice["precision"], cx=args_lattice["charges_x"], cy=args_lattice["charges_y"], sparse=args_lattice["sparse"])
#     return e, v

# def ground_state_Z2_exact(args_lattice, param):
#     energies_param = []
#     prec = args_lattice["precision"]
#     for p in param:
#         print(f"computing ground state for param: {p:.{prec}f}")
#         params = [args_lattice, p]
#         print(args_lattice["v0"])
#         energy, vectors = ground_state_Z2_exact_param(params=params)
#         energies_param.append(energy)
#         v0 = vectors[:,0]
#         args_lattice["v0"] = v0

#     return energies_param

# def ground_state_Z2_exact_test(args_lattice, param):
#     energies_param = []
#     prec = args_lattice["precision"]
#     for p in param:
#         print(f"computing ground state for param: {p:.{prec}f}")
#         Z2 = H_Z2_gauss(L=args_lattice["L"], l=args_lattice["l"], model=args_lattice["model"], U=args_lattice["U"], lamb=p)
#         if len(args_lattice["charges_x"]) > 0:
#             print("adding charges")
#             Z2.add_charges(rows=args_lattice["charges_x"], columns=args_lattice["charges_y"])
#             Z2._define_sector()
#         energy, vectors = Z2.diagonalize(v0=args_lattice["v0"], sparse=args_lattice["sparse"], path=args_lattice["path"], save=args_lattice["save"], precision=args_lattice["precision"], cx=args_lattice["charges_x"], cy=args_lattice["charges_y"])
#         energies_param.append(energy)
#         v0 = vectors[:,0]
#         args_lattice["v0"] = v0
#         print(args_lattice["v0"])

#     return energies_param

# def ground_state_Z2_param(params):
#     args_mps = params[0]
#     param = params[1]
#     ladder = MPS(
#         L=args_mps["L"],
#         d=args_mps["d"],
#         model=args_mps["model"],
#         chi=args_mps["chi"],
#         bc=args_mps["bc"],
#         h=param,
#     )
#     save = args_mps["save"]
#     precision = args_mps["precision"]
#     if ladder.model == "Z2_dual":
#         if args_mps["sector"] != "vacuum_sector":
#             ladder.Z2.add_charges(rows=args_mps["charges_x"], columns=args_mps["charges_y"])
#             ladder.Z2._define_sector()
#             print(ladder.Z2.charges, ladder.Z2.sector)
#     if args_mps["guess"] == []:
#         ladder._random_state(seed=3, chi=args_mps["chi"], type_shape=args_mps["type_shape"])
#         tensor_shapes(ladder.sites)
#         ladder.canonical_form(trunc_chi=True, trunc_tol=False)
#         if ladder.bc == "pbc":
#             a = np.zeros((1,2))
#             a[0,0] = 1
#             extra_ancillary_site = a.reshape((1,2,1))
#             ladder.sites.append(extra_ancillary_site)
#             ladder.L = len(ladder.sites)
#     else:
#         ladder.sites = args_mps["guess"].copy()
#         ladder.enlarge_chi()

#     energy, entropy, schmidt_vals, t_dmrg = ladder.DMRG(
#         trunc_tol=args_mps["trunc_tol"],
#         trunc_chi=args_mps["trunc_chi"],
#         where=args_mps["where"],
#         bond=args_mps["bond"],
#         n_sweeps=args_mps["n_sweeps"],
#         conv_tol=args_mps["conv_tol"]
#     )

#     print(f"energy of h:{param:.{precision}f}, L:{ladder.L} is:\n {energy}")
#     print(f"Schmidt values in the middle of the chain:\n {schmidt_vals}")
#     t_final = np.sum(t_dmrg)
#     t_final_gen = dt.timedelta(seconds=t_final)
#     # if t_final < 60:
#     #     t_unit = "sec(s)"
#     # elif t_final > 60 and t_final < 3600:
#     #     t_unit = "min(s)"
#     #     t_final = t_final/60
#     # elif t_final > 3600:
#     #     t_unit = "hour(s)"
#     #     t_final = t_final/3600

#     print(f"time of the whole search for h={param:.{precision}f} is: {t_final_gen}")

#     if not args_mps["training"]:
#         energy = energy[-1]

#     if save:
#         if ladder.bc == "pbc":
#             ladder.sites.pop()
#         ladder.L = len(ladder.sites)
#         ladder.save_sites(args_mps["path"], args_mps["precision"], args_mps["charges_x"], args_mps["charges_y"])
#     # args_mps["guess"] = ladder.sites.copy()
#     return energy, entropy, schmidt_vals, t_dmrg


def ground_state_Z2_multpr(args_mps, multpr_param, timeout=1000, cpu_percentage=90):
    max_workers = int(os.cpu_count() * (cpu_percentage / 100))
    timeout = timeout * len(multpr_param)
    stop_event = Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        args = [[args_mps, param] for param in multpr_param]
        try:
            results = executor.map(ground_state_Z2_param, args, timeout=timeout)
        except TimeoutError:
            print("Timeout reached! Requesting termination of tasks...")
            stop_event.set()
            results = None

        return results


def get_results(results):
    energies = []
    entropies = []
    schmidt_vals = []
    times = []
    # i = 0
    for result in results:
        # print(f"energy of h:{multpr_param[i]} is:\n {result[0][-1]}")
        energies.append(result[0])
        # print(f"entropies of h:{multpr_param[i]} is:\n {result[1]}")
        entropies.append(result[1])
        schmidt_vals.append(result[2])
        times.append(result[3])
        # i += 1
    return energies, entropies, schmidt_vals, times


def ground_state_Z2(args_mps, multpr, interval, reps: int = 1):
    if multpr:
        results_param = ground_state_Z2_multpr(args_mps=args_mps, multpr_param=interval)
        if results_param == None:
            return print("The computation exceeded the time limit")
        else:
            return get_results(results_param)
    else:
        energies, entropies, schmidt_vals, times = [], [], [], []
        for p in interval:
            print(f"\n*** Starting param: {p:.5f} in {dt.datetime.now()} ***\n")
            params = [args_mps, p]
            energy, entropy, schmidt_val, t_dmrg = ground_state_Z2_param(params)
            energies.append(energy)
            entropies.append(entropy)
            schmidt_vals.append(schmidt_val)
            times.append(t_dmrg)
        return energies, entropies, schmidt_vals, times

    # else:
    #     energies_param = []
    #     entropies_param = []
    #     schmidt_vals_param = []
    #     time_param = []
    #     threshold = 5
    #     slack = 1
    #     execution_times = []

    #     for p in param:
    #         params = [args_mps, p]
    #         for attempt in range(reps):
    #             with ThreadPoolExecutor() as executor:
    #                 try:
    #                     results = executor.map(ground_state_Z2_param, [params], timeout=threshold)
    #                     # future = executor.submit(ground_state_Z2_param, params=params)
    #                     # Attempt to execute within the threshold time
    #                     # results = future.result()
    #                     for result in results:
    #                         print("HHEEEEEEERRRRREEEEEEEE")
    #                         print(result)
    #                     energy, entropy, schmidt_vals, t_dmrg = results
    #                     print(f"Run for parameter: {p:.2f} attempt: {attempt} completed in {t_dmrg:.2f}s within threshold.")
    #                     execution_times.append(t_dmrg)
    #                     energies_param.append(energy)
    #                     entropies_param.append(entropy)
    #                     schmidt_vals_param.append(schmidt_vals)
    #                     time_param.append(t_dmrg)
    #                     break  # Exit retry loop since run was successful

    #                 except TimeoutError:
    #                     print(f"Run for parameter: {p:.2f} attempt: {attempt} exceeded threshold of {threshold:.2f}s. Retrying with random state...")
    #                     # Update parameters here as needed before retrying
    #                     args_mps["guess"] = []

    #             # # Ensure the executor is closed after each attempt
    #             # executor.shutdown(wait=True)

    #         # Update the threshold based on the average time with slack
    #         if execution_times:
    #             avg_time = sum(execution_times) / len(execution_times)
    #             threshold = avg_time * slack
    #             print(f"New threshold updated to {threshold:.2f}s")

    #     # for p in param:
    #     #     params = [args_mps, p]
    #     #     # Set the timeout period (in seconds)
    #     #     timeout_secs = new_timeout_secs # You can change this value according to your requirement
    #     #     # Set the alarm
    #     #     signal.alarm(int(timeout_secs+1))
    #     #     print(f"New timeout seconds: {int(timeout_secs+1)}")
    #     #     try:
    #     #         print("Running with guess state:")
    #     #         energy, entropy, schmidt_vals, t_dmrg = ground_state_Z2_param(params=params)

    #     #     except TimeoutError as e:
    #     #         print(e)
    #     #         args_mps["guess"] = []
    #     #         print("Running with random state:")
    #     #         energy, entropy, schmidt_vals, t_dmrg = ground_state_Z2_param(params=params)
    #     #     else:
    #     #         # Cancel the alarm if the algorithm finishes before the timeout
    #     #         signal.alarm(0)

    #     #     time_param.append(t_dmrg)
    #     #     t_mean = np.mean(time_param)
    #     #     t_std = np.std(time_param)
    #     #     new_timeout_secs = t_mean + 3*t_std
    #     #     energies_param.append(energy)
    #     #     entropies_param.append(entropy)
    #     #     schmidt_vals_param.append(schmidt_vals)
    #     #     time_param.append(t_dmrg)

    # return energies_param, entropies_param, schmidt_vals_param, time_param
