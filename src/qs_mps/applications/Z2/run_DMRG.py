import concurrent.futures
import multiprocessing
import numpy as np
import datetime as dt
from qs_mps.mps_class import MPS
from qs_mps. utils import tensor_shapes, get_precision

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
            ladder.Z2._define_sector()
    if args_mps["guess"] == []:
        print("Running with random state")
        ladder._random_state(seed=3, chi=args_mps["chi"], type_shape=args_mps["type_shape"])
        ladder.canonical_form(trunc_chi=True, trunc_tol=False)
        if ladder.bc == "pbc":
            a = np.zeros((1,2))
            a[0,0] = 1
            extra_ancillary_site = a.reshape((1,2,1))
            ladder.sites.append(extra_ancillary_site)
            ladder.L = len(ladder.sites)
    else:
        print("Running with guess state")
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
    t_final = np.sum(t_dmrg)
    t_final_gen = dt.timedelta(seconds=t_final)
    print(f"time of the whole search for h={param:.{precision}f} is: {t_final_gen}")
    
    if not args_mps["training"]:
        energy = energy[-1]

    if save:
        if ladder.bc == "pbc":
            ladder.sites.pop()
        ladder.L = len(ladder.sites)
        ladder.save_sites(args_mps["path"], args_mps["precision"], args_mps["charges_x"], args_mps["charges_y"])
    new_guess = ladder.sites.copy()
    return energy, entropy, schmidt_vals, t_dmrg, new_guess

def run_with_timeout(func, args, timeout):
    with multiprocessing.Pool() as pool:
        result = pool.apply_async(func, args)
        try:
            return result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            print(f"\n## TimeoutError: Algorithm exceeded {timeout} seconds ##\n")
            pool.terminate()  # Forcefully terminate the worker
            pool.join()
            return None  # Return None or any indicator of failure

def ground_state_Z2(args_mps, interval, reps=3):
    ene_tot = []
    ent_tot = []
    sm_tot = []
    t_tot = []
    timeout = 10000
    slack = 2
    params_not_found = []
    for p in interval:
        count_attempts = 0
        for attempt in range(reps):
            params = (args_mps, p)
            result = run_with_timeout(ground_state_Z2_param, (params,), timeout)
            if result is None:
                print(f"Computation for h={params[1]:.{args_mps["precision"]}f} timed out and was terminated.")
                args_mps["guess"] = []
                timeout = timeout * slack
                continue
            else:
                energy, entropy, sm, tdmrg, new_guess = result
                ene_tot.append(energy)
                ent_tot.append(entropy)
                sm_tot.append(sm)
                t_tot.append(tdmrg)
                args_mps["guess"] = new_guess.copy() 
                count_attempts = 1
                break
        
        if count_attempts == 0:
            print(f"h={params[1]:.{args_mps["precision"]}f}")
            params_not_found.append(params[1])
        if t_tot:
            avg_time = sum(t_tot) / len(t_tot)
            timeout = avg_time * slack
            print(f"New timeout updated to {timeout:.2f}s")
    
        print(f"Completed computation for h={params[1]:.{args_mps["precision"]}f}")
    print(f"Parameters not found are {len(params_not_found)}:\n{params_not_found}")
    
    return ene_tot, ent_tot, sm_tot, t_tot

l = 4
npoints = 10
h_i = 0.1
h_f = 1.0
L = 4
d = 2**l
chi = 50
type_shape = "rectangular"
model = "Z2_dual"
where = L//2
bond = True
path_tensor = f"C:/Users/HP/Desktop/projects/1_Z2"
save = False
interval = np.linspace(h_i, h_f, npoints)
num = (interval[-1] - interval[0]) / npoints
precision = get_precision(num)
charges_x = []
charges_y = []
if charges_x == []:
    sector = "vacuum_sector"
    charges_x = np.nan
    charges_y = np.nan
else:
    sector = f"{len(charges_x)}_particle(s)_sector"
number_sweeps = 2
conv_tol = 1e-10
training = True
init_tensor = []
boundcond = "obc"
args_mps = {
    "L": L,
    "d": d,
    "chi": chi,
    "type_shape": type_shape,
    "model": model,
    "trunc_tol": False,
    "trunc_chi": True,
    "where": where,
    "bond": bond,
    "path": path_tensor,
    "save": save,
    "precision": precision,
    "sector": sector,
    "charges_x": charges_x,
    "charges_y": charges_y,
    "n_sweeps": number_sweeps,
    "conv_tol": conv_tol,
    "training": training,
    "guess": init_tensor,
    "bc": boundcond,
}

if __name__ == "__main__":
    energy_chi, entropy_chi, schmidt_vals_chi, t_chi = ground_state_Z2(
        args_mps=args_mps, interval=interval
    )