import argparse
import numpy as np
from qs_mps.mps_class import MPS
from qs_mps.utils import get_precision, save_list_of_lists, access_txt, swap_rows
from qs_mps.gs_multiprocessing import ground_state_ising

import signal
import time

# Define a function to handle the timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Algorithm took too long to execute")

# Set the signal handler
signal.signal(signal.SIGALRM, timeout_handler)

# DENSITY MATRIX RENORMALIZATION GROUP to find ground states of the 
# 1D Ising Transverse Field model changing the transverse field parameters

parser = argparse.ArgumentParser(prog="gs_search_Ising")
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field)", type=float
)
parser.add_argument(
    "h_f", help="Final value of h (external transverse field)", type=float
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("-L", "--Ls", help="Spin chain lengths", nargs="*", type=int)
parser.add_argument("-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-d","--dimension", help="Physical dimension. By default 2", default=2, type=int)
parser.add_argument(
    "-ty", "--type_shape", help="Type of shape of the bond dimension. Available are: 'trapezoidal', 'pyramidal', 'rectangular'", default="trapezoidal", type=str
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Ising", type=str
)
parser.add_argument(
    "-s",
    "--number_sweeps",
    help="Number of sweeps during the compression algorithm for each trotter step",
    default=8,
    type=int,
)
parser.add_argument(
    "-cv",
    "--conv_tol",
    help="Convergence tolerance of the compression algorithm",
    default=1e-10,
    type=float,
)
parser.add_argument(
    "-fid",
    "--fidelity",
    help="Fidelity with exact solution. Doable only for small Ising chains",
    action="store_true",
)
parser.add_argument(
    "-b",
    "--bond",
    help="Save the schmidt values for one bond. If False save for each bond. By default True",
    action="store_false",
)
parser.add_argument(
    "-w",
    "--where",
    help="Bond where we want to observe the Schmidt values, should be between 1 and (L-1)",
    default=-1,
    type=int,
)
parser.add_argument(
    "-tr",
    "--training",
    help="Save all the energies during the variational optimization. By default True",
    action="store_false",
)

args = parser.parse_args()

# define the interval of equally spaced values of external field
interval_hx = np.linspace(args.h_i, args.h_f, args.npoints).tolist()
interval_hy = np.linspace(args.h_i, args.h_f, args.npoints)
interval_hx.reverse()

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/0_ISING"
    path_tensor = "D:/code/projects/0_ISING"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/0_ISING"
    path_tensor = "/Users/fradm98/Desktop/projects/0_ISING"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/0_ISING"
    path_tensor = "/Users/fradm/Desktop/projects/0_ISING"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

path = f"{parent_path}/results/tensors"
num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num)


# ---------------------------------------------------------
# DMRG
# ---------------------------------------------------------
for L in args.Ls:
    if args.where == -1:
        args.where = L // 2
    elif args.where == -2:
        args.bond = False
    for chi in args.chis:  # L // 2 + 1
        # init_state = np.zeros((1, 2, 1))
        # init_state[0, 0, 0] = 1
        # init_tensor = [init_state for _ in range(L)]
        energy_chi = []
        entropy_chi = []
        schmidt_vals_chi = []
        time_chi = []
        exceptions_chi = np.zeros((len(interval_hy),len(interval_hx)))
        for idx0, J in enumerate(interval_hy):
            energy_J = []
            entropy_J = []
            schmidt_vals_J = []
            t_slice = []
            new_timeout_secs = 5
            for idx1, h in enumerate(interval_hx):
                chain = MPS(
                    L=L,
                    d=2,
                    model="Ising",
                    chi=chi,
                    h=h,
                    J=J,
                    eps=1e-5,
                )                
                if h == interval_hx[0]:
                    # init_tensor = [init_state for _ in range(L)]
                    # chain.sites = init_tensor.copy()
                    # chain.enlarge_chi(type_shape="rectangular", prnt=False)
                    chain._random_state(3, chi=chi)
                    chain.canonical_form(trunc_chi=False, trunc_tol=True)
                else:
                    chain.sites = init_tensor.copy()

                # Set the timeout period (in seconds)
                timeout_secs = new_timeout_secs # You can change this value according to your requirement

                # Set the alarm
                signal.alarm(int(timeout_secs+1))
                print(f"New timeout seconds: {int(timeout_secs+1)}")
                try:
                    # Call your algorithm function with the initial parameter
                    print("Running with guess state:")
                    energy, entropy, schmidt_vals, t_dmrg = chain.DMRG(
                        trunc_tol=False,
                        trunc_chi=True,
                        where=args.where,
                        bond=args.bond,
                    )
                except TimeoutError as e:
                    print(e)
                    # Modify the parameter and call the algorithm again
                    chain._random_state(seed=7, type_shape="rectangular", chi=chi)
                    chain.canonical_form()
                    print("Running with random state:")
                    energy, entropy, schmidt_vals, t_dmrg = chain.DMRG(
                        trunc_tol=False,
                        trunc_chi=True,
                        where=args.where,
                        bond=args.bond,
                    )
                    exceptions_chi[idx0,idx1] = 1
                else:
                    # Cancel the alarm if the algorithm finishes before the timeout
                    signal.alarm(0)

                t_slice.append(t_dmrg)
                t_mean = np.mean(t_slice)
                t_std = np.std(t_slice)
                new_timeout_secs = t_mean + 3*t_std
                
                print(f"energy of h:{h:.{precision}f}, J:{J:.{precision}f} is:\n {energy}")
                print(f"Schmidt values in the middle of the chain:\n {schmidt_vals}")

                chain.save_sites(path=path_tensor, precision=precision)
                if args.training:
                    energy_J.append(energy)
                else:
                    energy_J.append(energy[-1])
                entropy_J.append(entropy)
                schmidt_vals_J.append(schmidt_vals)

                if h == interval_hx[0]:
                    init_tensor_J = chain.sites.copy()
                init_tensor = chain.sites.copy()
            
            init_tensor = init_tensor_J.copy()

            time_chi.append(t_slice)
            energy_chi.append(energy_J)
            entropy_chi.append(entropy_J)
            schmidt_vals_chi.append(schmidt_vals_J)
        # energy_chi, entropy_chi, schmidt_vals_chi = ground_state_ising(
        #     args_mps=args_mps, multpr=False, param=interval
        # )

        if args.bond == False:
            args.where = "all"

        np.save(f"{parent_path}/results/energy_data/", exceptions_chi)

        if args.training:
            energy_chi = np.swapaxes(swap_rows(np.asarray(energy_chi)), axis1=0, axis2=1)
            np.save(
                f"{parent_path}/results/energy_data/energies_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_J_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                energy_chi,
            )
            energy_min = []
            for i in range(len(interval_hx)):
                energy_min_i = []
                for j in range(len(interval_hy)):
                    energy_min_i.append(energy_chi[i][j][-1])
                energy_min.append(energy_min_i)
            np.save(f"{parent_path}/results/energy_data/energy_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_J_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}", energy_min)
        else:
            energy_chi = np.array(energy_chi)
            energy_chi = np.array_split(energy_chi, args.npoints)
            save_list_of_lists(
                f"{parent_path}/results/energy_data/energies_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                energy_chi,
            )
        entropy_chi = np.array(entropy_chi)
        entropy_chi = np.array_split(entropy_chi, args.npoints)
        save_list_of_lists(
            f"{parent_path}/results/entropy/{args.where}_bond_entropy_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
            entropy_chi,
        )
        schmidt_vals_chi = np.array(schmidt_vals_chi)
        schmidt_vals_chi = np.array_split(schmidt_vals_chi, args.npoints)
        save_list_of_lists(
            f"{parent_path}/results/entropy/{args.where}_schmidt_values_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
            schmidt_vals_chi,
        )
        if args.where == "all":
            entropy_mid = access_txt(
                f"{parent_path}/results/entropy/{args.where}_bond_entropy_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                L // 2,
            )
            np.savetxt(
                f"{parent_path}/results/entropy/{L // 2}_bond_entropy_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                entropy_mid,
            )
